//! Codegen: AST verificado → LLVM IR, e execução via JIT.
//!
//! F1:  expressão única → função `top()` → JIT
//! F2b: programa inteiro → funções, variáveis, chamadas, `main`
//! F2c: structs (LLVM agregados + GEP), receiver como param 0,
//!      member access (`p.x`) e chamadas de método (`p.len_sq()`)
//! F4:  int (i64) e float (f64) separados, bool (i1), strings internadas
//!      (global única por conteúdo — ==/!= vira comparação de ponteiro),
//!      casts inseridos pelo type checker (`Cast`)
//!
//! Entrada: o AST já verificado pelo type checker (F4) — os tipos estão
//! coerentes (casts implícitos inseridos) e os erros de tipo já foram
//! reportados. As verificações restantes aqui são defensivas.

use crate::ast::*;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::module::Module;
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, StructType};
use inkwell::values::{
    BasicMetadataValueEnum, BasicValueEnum, FloatValue, FunctionValue, IntValue, PointerValue,
};
use inkwell::{FloatPredicate, IntPredicate, OptimizationLevel};
use std::collections::HashMap;

/// Assinatura da função top-level do caminho F1 (expressão).
pub type TopFn = unsafe extern "C" fn() -> f64;

/// Resultado da execução de `main`.
pub enum MainResult {
    Float(f64),
    Int(i64),
    Void,
}

/// Definição de struct resolvida: campos na ordem declarada.
struct StructDef {
    fields: Vec<(String, Type)>,
}

impl StructDef {
    fn index(&self, name: &str) -> Option<u32> {
        self.fields.iter().position(|(n, _)| n == name).map(|i| i as u32)
    }
}

/// Valor de uma expressão + seu tipo.
struct Typed<'ctx> {
    value: BasicValueEnum<'ctx>,
    ty: Type,
}

pub struct Codegen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    /// Símbolo da função → tipo de retorno (None = void).
    ret_types: HashMap<String, Option<Type>>,
    /// Variáveis locais da função corrente: nome → (tipo, alloca).
    vars: HashMap<String, (Type, PointerValue<'ctx>)>,
    /// Structs declaradas: nome → (tipo LLVM, definição).
    structs: HashMap<String, (StructType<'ctx>, StructDef)>,
    /// Enums declarados: nome → variante → valor (sequencial desde 0).
    /// Internamente são i64 (F4).
    enums: HashMap<String, HashMap<String, i64>>,
    /// Strings internadas: conteúdo → global constante (uma por string
    /// única — ==/!= vira comparação de ponteiro).
    strings: HashMap<String, PointerValue<'ctx>>,
    /// Tipo de retorno da função sendo gerada (None = void).
    current_ret: Option<Type>,
    /// Função sendo gerada (para criar basic blocks).
    current_fn: Option<FunctionValue<'ctx>>,
    /// O bloco atual terminou com return/break/continue?
    block_open: bool,
    /// Pilha de loops: (alvo do continue, alvo do break).
    loop_stack: Vec<(BasicBlock<'ctx>, BasicBlock<'ctx>)>,
}

impl<'ctx> Codegen<'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
        let module = context.create_module("forge");
        let builder = context.create_builder();
        Self {
            context,
            module,
            builder,
            ret_types: HashMap::new(),
            vars: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            strings: HashMap::new(),
            current_ret: None,
            current_fn: None,
            block_open: true,
            loop_stack: Vec::new(),
        }
    }

    // ======================= tipos e helpers =========================

    fn is_float(ty: &Type) -> bool {
        matches!(ty, Type::Float)
    }

    fn is_int(ty: &Type) -> bool {
        matches!(ty, Type::Int)
    }

    /// Tipo LLVM correspondente a um tipo da linguagem.
    fn llvm_type(&self, ty: &Type) -> Result<BasicTypeEnum<'ctx>, String> {
        match ty {
            Type::Float => Ok(self.context.f64_type().into()),
            Type::Int => Ok(self.context.i64_type().into()),
            Type::Bool => Ok(self.context.bool_type().into()),
            // string = ponteiro para global interna (F4).
            Type::Str => Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
            // enums são i64 internamente (F4).
            Type::Named(n) if self.enums.contains_key(n) => Ok(self.context.i64_type().into()),
            Type::Named(n) => self
                .structs
                .get(n)
                .map(|(st, _)| (*st).into())
                .ok_or_else(|| format!("tipo '{n}' desconhecido (struct não declarada ou tipo não suportado)")),
            Type::Void => Err("tipo 'void' não é um valor".into()),
        }
    }

    /// Extrai FloatValue, com erro claro se o tipo não é float.
    fn as_float(&self, t: &Typed<'ctx>, what: &str) -> Result<FloatValue<'ctx>, String> {
        if !Self::is_float(&t.ty) {
            return Err(format!(
                "'{what}' deve ser float, encontrou '{}'",
                type_name(&t.ty)
            ));
        }
        Ok(t.value.into_float_value())
    }

    /// Extrai IntValue (i64) de um int/enum, com erro claro.
    fn as_int(&self, t: &Typed<'ctx>, what: &str) -> Result<IntValue<'ctx>, String> {
        if !Self::is_int(&t.ty) && !self.enums.contains_key(&type_name(&t.ty)) {
            return Err(format!(
                "'{what}' deve ser int, encontrou '{}'",
                type_name(&t.ty)
            ));
        }
        Ok(t.value.into_int_value())
    }

    // ========================= F2c: programa ==========================

    /// Compila um programa: enums → structs → cabeçalhos → corpos.
    pub fn compile_program(&mut self, program: &Program) -> Result<(), String> {
        self.declare_enums(program)?;
        self.declare_structs(program)?;
        for decl in &program.decls {
            if let Decl::Func(f) = decl {
                self.declare_func(f)?;
            }
        }
        for decl in &program.decls {
            if let Decl::Func(f) = decl {
                self.gen_func(f)?;
            }
        }
        Ok(())
    }

    /// Passo 0: enums viram mapas de constantes (valores 0, 1, 2...
    /// como i64 — F4 trocou o float interno por int real).
    fn declare_enums(&mut self, program: &Program) -> Result<(), String> {
        for decl in &program.decls {
            if let Decl::Enum(e) = decl {
                let mut variants = HashMap::new();
                for (i, v) in e.variants.iter().enumerate() {
                    variants.insert(v.clone(), i as i64);
                }
                self.enums.insert(e.name.clone(), variants);
            }
        }
        Ok(())
    }

    /// Passo 1: cria os tipos LLVM dos structs, valida campos e
    /// detecta ciclos (struct recursiva por valor é inválida).
    fn declare_structs(&mut self, program: &Program) -> Result<(), String> {
        // 1. Coleta campos por struct (sem LLVM ainda).
        let mut raw: HashMap<String, Vec<(String, Type)>> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        for decl in &program.decls {
            if let Decl::Struct(s) = decl {
                let fields = s.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect();
                order.push(s.name.clone());
                raw.insert(s.name.clone(), fields);
            }
        }
        // 2. Valida: cada campo é primitivo | enum | struct declarada.
        for (name, fields) in &raw {
            for (fname, ty) in fields {
                let valid = match ty {
                    Type::Int | Type::Float | Type::Bool | Type::Str => true,
                    Type::Named(n) => self.enums.contains_key(n) || raw.contains_key(n),
                    _ => false,
                };
                if !valid {
                    return Err(format!(
                        "struct '{name}': campo '{fname}' de tipo '{}' desconhecido",
                        type_name(ty)
                    ));
                }
            }
        }
        // 3. Detecta ciclos de struct por valor (a→b→a).
        for name in &order {
            let mut path: Vec<String> = Vec::new();
            if let Some(cycle) = Self::struct_cycle(&raw, name, &mut path) {
                return Err(format!(
                    "struct recursiva por valor não suportada (F5: ponteiros): {cycle}"
                ));
            }
        }
        // 4. Cria todos os tipos (opaque), depois preenche corpos —
        //    qualquer ordem de referência entre structs funciona.
        for name in &order {
            let st = self.context.opaque_struct_type(name);
            self.structs.insert(name.clone(), (st, StructDef { fields: Vec::new() }));
        }
        for name in &order {
            let fields = &raw[name];
            let field_tys: Vec<BasicTypeEnum> = fields
                .iter()
                .map(|(_, t)| self.llvm_type(t).expect("validado no passo 2"))
                .collect();
            let (st, def) = self.structs.get_mut(name).expect("inserido no passo 4");
            st.set_body(&field_tys, false);
            def.fields = fields.clone();
        }
        Ok(())
    }

    /// DFS: retorna a descrição do ciclo se `start` se contém (por valor).
    fn struct_cycle(
        raw: &HashMap<String, Vec<(String, Type)>>,
        start: &str,
        path: &mut Vec<String>,
    ) -> Option<String> {
        if let Some(pos) = path.iter().position(|p| p == start) {
            let mut cycle: Vec<&str> = path[pos..].iter().map(|s| s.as_str()).collect();
            cycle.push(start);
            return Some(cycle.join(" → "));
        }
        let fields = raw.get(start)?;
        path.push(start.to_string());
        for (_, ty) in fields {
            let tn = type_name(ty);
            if raw.contains_key(&tn) {
                if let Some(cycle) = Self::struct_cycle(raw, &tn, path) {
                    return Some(cycle);
                }
            }
        }
        path.pop();
        None
    }

    /// Símbolo da função no módulo. Métodos (receiver) ganham prefixo do
    /// tipo para evitar colisão: `Point.len_sq`.
    fn func_symbol(f: &FuncDecl) -> String {
        match &f.receiver {
            Some(r) => format!("{}.{}", type_name(&r.ty), f.name),
            None => f.name.clone(),
        }
    }

    /// Cria o cabeçalho da função no módulo (sem corpo).
    fn declare_func(&mut self, f: &FuncDecl) -> Result<(), String> {
        let void = self.context.void_type();

        // Params: receiver (se houver) + params declarados.
        // Receiver é passado por REFERÊNCIA (ponteiro) — métodos mutam
        // o objeto original, como `this`/`&mut self`. Params por valor.
        let mut param_tys: Vec<BasicMetadataTypeEnum> = Vec::new();
        if let Some(_r) = &f.receiver {
            param_tys.push(
                self.context
                    .ptr_type(inkwell::AddressSpace::default())
                    .into(),
            );
        }
        for p in &f.params {
            param_tys.push(self.llvm_type(&p.ty)?.into());
        }

        let ret_ty = match &f.ret {
            None => None,
            Some(t) => Some(t.clone()),
        };
        let fn_ty = match &ret_ty {
            Some(t) => self.llvm_type(t)?.fn_type(&param_tys, false),
            None => void.fn_type(&param_tys, false),
        };
        let symbol = Self::func_symbol(f);
        self.ret_types.insert(symbol.clone(), ret_ty);
        self.module.add_function(&symbol, fn_ty, None);
        Ok(())
    }

    /// Gera o corpo de uma função já declarada.
    fn gen_func(&mut self, f: &FuncDecl) -> Result<(), String> {
        let symbol = Self::func_symbol(f);
        let function = self
            .module
            .get_function(&symbol)
            .ok_or_else(|| format!("função '{symbol}' declarada mas não encontrada"))?;
        self.vars.clear();
        self.current_ret = f.ret.clone();
        self.current_fn = Some(function);
        self.loop_stack.clear();

        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);
        self.block_open = true;

        // Params viram variáveis locais (alloca + store) — padrão Kaleidoscope.
        // O receiver é diferente: já É um ponteiro, então a variável local
        // aponta direto para o objeto original (mutações persistem).
        let mut idx = 0u32;
        if let Some(r) = &f.receiver {
            let p = function
                .get_nth_param(idx)
                .ok_or_else(|| "parametro receiver faltando".to_string())?;
            let ptr = p.into_pointer_value();
            self.vars.insert(r.name.clone(), (r.ty.clone(), ptr));
            idx += 1;
        }
        for p in &f.params {
            let param = function
                .get_nth_param(idx)
                .ok_or_else(|| format!("parametro '{}' faltando", p.name))?;
            self.assign_param(&p.name, &p.ty, param)?;
            idx += 1;
        }

        // Corpo. Se um statement fechou o bloco (return/break/continue),
        // o resto é inalcançável.
        for stmt in &f.body {
            if !self.block_open {
                break;
            }
            self.gen_stmt(stmt)?;
        }

        // Retorno implícito: 0 para int/float, void caso contrário.
        if self.block_open {
            match &f.ret {
                Some(t) if Self::is_float(t) => {
                    let zero = self.context.f64_type().const_float(0.0);
                    self.builder.build_return(Some(&zero)).map_err(|e| e.to_string())?;
                }
                Some(t) if Self::is_int(t) => {
                    let zero = self.context.i64_type().const_zero();
                    self.builder.build_return(Some(&zero)).map_err(|e| e.to_string())?;
                }
                Some(_) => {
                    return Err(format!(
                        "função '{symbol}' retorna struct mas não tem return explícito (structs como retorno implícito: F5)"
                    ));
                }
                None => {
                    self.builder.build_return(None).map_err(|e| e.to_string())?;
                }
            }
        }
        self.current_fn = None;
        Ok(())
    }

    fn assign_param(
        &mut self,
        name: &str,
        ty: &Type,
        param: BasicValueEnum<'ctx>,
    ) -> Result<(), String> {
        let ll = self.llvm_type(ty)?;
        let ptr = self.builder.build_alloca(ll, name).map_err(|e| e.to_string())?;
        self.builder.build_store(ptr, param).map_err(|e| e.to_string())?;
        self.vars.insert(name.to_string(), (ty.clone(), ptr));
        Ok(())
    }

    // ----------------------- statements -------------------------------

    fn gen_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::Let { name, ty, value, .. } => {
                let val = self.gen_expr(value)?;
                if let Some(annot) = ty {
                    if type_name(annot) != type_name(&val.ty) {
                        return Err(format!(
                            "tipo anotado '{0}' não bate com o valor '{1}' (inferência real: F4)",
                            type_name(annot),
                            type_name(&val.ty)
                        ));
                    }
                }
                let ll = self.llvm_type(&val.ty)?;
                let ptr = self
                    .builder
                    .build_alloca(ll, name)
                    .map_err(|e| e.to_string())?;
                self.builder.build_store(ptr, val.value).map_err(|e| e.to_string())?;
                self.vars.insert(name.clone(), (val.ty.clone(), ptr));
                Ok(())
            }
            Stmt::Return(value, _) => {
                match value {
                    Some(v) => {
                        let val = self.gen_expr(v)?;
                        if let Some(expected) = &self.current_ret {
                            if type_name(expected) != type_name(&val.ty) {
                                return Err(format!(
                                    "retorno '{}' não bate com o tipo declarado '{}'",
                                    type_name(&val.ty),
                                    type_name(expected)
                                ));
                            }
                        }
                        self.builder
                            .build_return(Some(&val.value))
                            .map_err(|e| e.to_string())?;
                    }
                    None => {
                        self.builder.build_return(None).map_err(|e| e.to_string())?;
                    }
                }
                self.block_open = false;
                Ok(())
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                let function = self
                    .current_fn
                    .ok_or("if fora de função")?;
                let cond_val = self.gen_cond(cond)?;
                let then_bb = self.context.append_basic_block(function, "then");
                let else_bb = self.context.append_basic_block(function, "else");
                let merge_bb = self.context.append_basic_block(function, "merge");

                self.builder
                    .build_conditional_branch(cond_val, then_bb, else_bb)
                    .map_err(|e| e.to_string())?;

                // ramo then
                self.builder.position_at_end(then_bb);
                self.block_open = true;
                for s in then_body {
                    if !self.block_open {
                        break;
                    }
                    self.gen_stmt(s)?;
                }
                if self.block_open {
                    self.builder
                        .build_unconditional_branch(merge_bb)
                        .map_err(|e| e.to_string())?;
                }

                // ramo else
                self.builder.position_at_end(else_bb);
                self.block_open = true;
                if let Some(els) = else_body {
                    for s in els {
                        if !self.block_open {
                            break;
                        }
                        self.gen_stmt(s)?;
                    }
                }
                if self.block_open {
                    self.builder
                        .build_unconditional_branch(merge_bb)
                        .map_err(|e| e.to_string())?;
                }

                // continua no merge (se algum ramo chegou lá)
                self.builder.position_at_end(merge_bb);
                self.block_open = true;
                Ok(())
            }
            Stmt::For { init, cond, post, body, .. } => {
                let function = self
                    .current_fn
                    .ok_or("for fora de função")?;
                if let Some(init) = init {
                    self.gen_stmt(init.as_ref())?;
                }

                let cond_bb = self.context.append_basic_block(function, "for.cond");
                let body_bb = self.context.append_basic_block(function, "for.body");
                let post_bb = self.context.append_basic_block(function, "for.post");
                let end_bb = self.context.append_basic_block(function, "for.end");

                self.builder
                    .build_unconditional_branch(cond_bb)
                    .map_err(|e| e.to_string())?;

                // condição (ou branch direto se não houver)
                self.builder.position_at_end(cond_bb);
                self.block_open = true;
                if let Some(cond) = cond {
                    let c = self.gen_cond(cond)?;
                    self.builder
                        .build_conditional_branch(c, body_bb, end_bb)
                        .map_err(|e| e.to_string())?;
                } else {
                    self.builder
                        .build_unconditional_branch(body_bb)
                        .map_err(|e| e.to_string())?;
                }

                // corpo
                self.builder.position_at_end(body_bb);
                self.block_open = true;
                self.loop_stack.push((post_bb, end_bb));
                for s in body {
                    if !self.block_open {
                        break;
                    }
                    self.gen_stmt(s)?;
                }
                if self.block_open {
                    self.builder
                        .build_unconditional_branch(post_bb)
                        .map_err(|e| e.to_string())?;
                }
                self.loop_stack.pop();

                // post
                self.builder.position_at_end(post_bb);
                self.block_open = true;
                if let Some(post) = post {
                    self.gen_stmt(post.as_ref())?;
                }
                if self.block_open {
                    self.builder
                        .build_unconditional_branch(cond_bb)
                        .map_err(|e| e.to_string())?;
                }

                // fim do loop
                self.builder.position_at_end(end_bb);
                self.block_open = true;
                Ok(())
            }
            Stmt::Break(_) => {
                let (_, break_bb) = self
                    .loop_stack
                    .last()
                    .ok_or("break fora de loop")?;
                self.builder
                    .build_unconditional_branch(*break_bb)
                    .map_err(|e| e.to_string())?;
                self.block_open = false;
                Ok(())
            }
            Stmt::Continue(_) => {
                let (continue_bb, _) = self
                    .loop_stack
                    .last()
                    .ok_or("continue fora de loop")?;
                self.builder
                    .build_unconditional_branch(*continue_bb)
                    .map_err(|e| e.to_string())?;
                self.block_open = false;
                Ok(())
            }
            Stmt::Assign { target, value, .. } => {
                let val = self.gen_expr(value)?;
                match self.gen_lvalue(target)? {
                    Some((tty, ptr)) => {
                        if type_name(&tty) != type_name(&val.ty) {
                            return Err(format!(
                                "atribuição: alvo '{}' não aceita valor '{}'",
                                type_name(&tty),
                                type_name(&val.ty)
                            ));
                        }
                        self.builder
                            .build_store(ptr, val.value)
                            .map_err(|e| e.to_string())?;
                        Ok(())
                    }
                    None => Err(
                        "alvo de atribuição deve ser variável ou campo de variável".into()
                    ),
                }
            }
            Stmt::Expr(e) => {
                // Chamada como statement: o valor é descartado, então
                // funções void são permitidas aqui.
                if let Expr::Call { callee, args, .. } = e {
                    let (function, arg_vals, symbol) = self.resolve_call(callee, args)?;
                    self.builder
                        .build_call(function, &arg_vals, "calltmp")
                        .map_err(|e| format!("{symbol}: {e}"))?;
                } else {
                    self.gen_expr(e)?;
                }
                Ok(())
            }
        }
    }

    /// Resolve o ponteiro de escrita de um lvalue:
    ///   - variável (Ident) → alloca da variável
    ///   - campo (Member) → GEP encadeado a partir do lvalue do obj
    ///   - temporários → `None` (não são endereçáveis)
    fn gen_lvalue(&self, expr: &Expr) -> Result<Option<(Type, PointerValue<'ctx>)>, String> {
        match expr {
            Expr::Ident(name, _) => self
                .vars
                .get(name)
                .map(|(ty, ptr)| Some((ty.clone(), *ptr)))
                .ok_or_else(|| format!("variável '{name}' não definida")),
            Expr::Member { obj, field, .. } => {
                let Some((oty, optr)) = self.gen_lvalue(obj)? else {
                    return Ok(None);
                };
                let ty_name = type_name(&oty);
                let (st, def) = match self.structs.get(&ty_name) {
                    Some(x) => x,
                    None => return Ok(None), // enum.field não é lvalue
                };
                let idx = def.index(field).ok_or_else(|| {
                    format!("struct '{ty_name}' não tem campo '{field}'")
                })?;
                let (_, fty) = &def.fields[idx as usize];
                let ptr = self
                    .builder
                    .build_struct_gep(*st, optr, idx, "gep")
                    .map_err(|e| e.to_string())?;
                Ok(Some((fty.clone(), ptr)))
            }
            _ => Ok(None),
        }
    }

    // ----------------------- expressões -------------------------------

    fn gen_expr(&mut self, expr: &Expr) -> Result<Typed<'ctx>, String> {
        match expr {
            Expr::Int(n, _) => Ok(Typed {
                value: self.context.i64_type().const_int(*n as u64, true).into(),
                ty: Type::Int,
            }),
            Expr::Float(n, _) => Ok(Typed {
                value: self.context.f64_type().const_float(*n).into(),
                ty: Type::Float,
            }),
            Expr::Bool(b, _) => Ok(Typed {
                value: self.context.bool_type().const_int(*b as u64, false).into(),
                ty: Type::Bool,
            }),
            Expr::Str(s, _) => {
                let ptr = self.intern_string(s)?;
                Ok(Typed { value: ptr.into(), ty: Type::Str })
            }
            Expr::Ident(name, _) => {
                let (ty, ptr) = self
                    .vars
                    .get(name)
                    .ok_or_else(|| format!("variável '{name}' não definida"))?;
                let ll = self.llvm_type(ty)?;
                let v = self
                    .builder
                    .build_load(ll, *ptr, name)
                    .map_err(|e| e.to_string())?;
                Ok(Typed { value: v, ty: ty.clone() })
            }
            Expr::Binary { op, lhs, rhs, .. } => self.gen_binary(*op, lhs, rhs),
            Expr::StructLit { name, fields, .. } => {
                let (st, def) = self
                    .structs
                    .get(name)
                    .ok_or_else(|| format!("struct '{name}' não declarada"))?;
                let st = *st;
                // Clona os campos: gen_expr precisa de &mut self.
                let field_defs = def.fields.clone();
                // Materializa num alloca temporário e carrega como valor.
                let temp = self
                    .builder
                    .build_alloca(st, name)
                    .map_err(|e| e.to_string())?;
                let mut missing: Vec<String> =
                    field_defs.iter().map(|(n, _)| n.clone()).collect();
                for (fname, fexpr) in fields {
                    let Some(idx) = field_defs.iter().position(|(n, _)| n == fname) else {
                        return Err(format!("struct '{name}' não tem campo '{fname}'"));
                    };
                    let v = self.gen_expr(fexpr)?;
                    let ptr = self
                        .builder
                        .build_struct_gep(st, temp, idx as u32, "gep")
                        .map_err(|e| e.to_string())?;
                    self.builder.build_store(ptr, v.value).map_err(|e| e.to_string())?;
                    missing.retain(|n| n != fname);
                }
                if let Some(fname) = missing.first() {
                    return Err(format!("struct literal '{name}' sem o campo '{fname}'"));
                }
                let val = self
                    .builder
                    .build_load(st, temp, "structval")
                    .map_err(|e| e.to_string())?;
                Ok(Typed { value: val, ty: Type::Named(name.clone()) })
            }
            Expr::Member { obj, field, .. } => {
                // Enum value: `Mood.Stressed` — obj é um Ident de enum.
                if let Expr::Ident(enum_name, _) = obj.as_ref() {
                    if let Some(variants) = self.enums.get(enum_name) {
                        let value = variants.get(field).ok_or_else(|| {
                            format!("enum '{enum_name}' não tem variante '{field}'")
                        })?;
                        return Ok(Typed {
                            value: self.context.i64_type().const_int(*value as u64, true).into(),
                            ty: Type::Named(enum_name.clone()),
                        });
                    }
                }
                // Struct member: se o obj é lvalue, GEP direto no ponteiro
                // (mais eficiente que materializar temp).
                if let Some((oty, optr)) = self.gen_lvalue(obj)? {
                    let ty_name = type_name(&oty);
                    let (st, def) = self
                        .structs
                        .get(&ty_name)
                        .ok_or_else(|| format!("tipo '{ty_name}' não é struct"))?;
                    let idx = def.index(field).ok_or_else(|| {
                        format!("struct '{ty_name}' não tem campo '{field}'")
                    })?;
                    let (_, fty) = &def.fields[idx as usize];
                    let ptr = self
                        .builder
                        .build_struct_gep(*st, optr, idx, "gep")
                        .map_err(|e| e.to_string())?;
                    let ll = self.llvm_type(fty)?;
                    let v = self
                        .builder
                        .build_load(ll, ptr, "memberval")
                        .map_err(|e| e.to_string())?;
                    Ok(Typed { value: v, ty: fty.clone() })
                } else {
                    // Temporário: materializa num alloca e faz GEP.
                    let obj_v = self.gen_expr(obj)?;
                    let ty_name = type_name(&obj_v.ty);
                    let (st, def) = self
                        .structs
                        .get(&ty_name)
                        .ok_or_else(|| format!("tipo '{ty_name}' não é struct"))?;
                    let idx = def.index(field).ok_or_else(|| {
                        format!("struct '{ty_name}' não tem campo '{field}'")
                    })?;
                    let (_, fty) = &def.fields[idx as usize];
                    let temp = self
                        .builder
                        .build_alloca(*st, "membertmp")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_store(temp, obj_v.value)
                        .map_err(|e| e.to_string())?;
                    let ptr = self
                        .builder
                        .build_struct_gep(*st, temp, idx, "gep")
                        .map_err(|e| e.to_string())?;
                    let ll = self.llvm_type(fty)?;
                    let v = self
                        .builder
                        .build_load(ll, ptr, "memberval")
                        .map_err(|e| e.to_string())?;
                    Ok(Typed { value: v, ty: fty.clone() })
                }
            }
            Expr::Call { callee, args, .. } => {
                let (function, arg_vals, symbol) = self.resolve_call(callee, args)?;
                self.finish_call(function, &arg_vals, &symbol)
            }
            Expr::Unary { op, operand, .. } => match op {
                UnOp::Neg => {
                    let v = self.gen_expr(operand)?;
                    match v.ty {
                        Type::Float => {
                            let f = self.as_float(&v, "operando")?;
                            let r = self
                                .builder
                                .build_float_neg(f, "negtmp")
                                .map_err(|e| e.to_string())?;
                            Ok(Typed { value: r.into(), ty: Type::Float })
                        }
                        Type::Int => {
                            let i = self.as_int(&v, "operando")?;
                            let r = self
                                .builder
                                .build_int_neg(i, "negtmp")
                                .map_err(|e| e.to_string())?;
                            Ok(Typed { value: r.into(), ty: Type::Int })
                        }
                        other => Err(format!(
                            "'-' exige int ou float, encontrou '{}'",
                            type_name(&other)
                        )),
                    }
                }
                UnOp::Not => {
                    let v = self.gen_expr(operand)?;
                    let v = self.as_bool(&v, "operando")?;
                    let r = self
                        .builder
                        .build_not(v, "nottmp")
                        .map_err(|e| e.to_string())?;
                    Ok(Typed {
                        value: r.into(),
                        ty: Type::Bool,
                    })
                }
            },
            Expr::Cast { to, expr, .. } => {
                let v = self.gen_expr(expr)?;
                match (to, &v.ty) {
                    (Type::Float, Type::Int) => {
                        let f = self
                            .builder
                            .build_signed_int_to_float(v.value.into_int_value(), self.context.f64_type(), "itof")
                            .map_err(|e| e.to_string())?;
                        Ok(Typed { value: f.into(), ty: Type::Float })
                    }
                    (Type::Int, Type::Float) => {
                        let i = self
                            .builder
                            .build_float_to_signed_int(v.value.into_float_value(), self.context.i64_type(), "ftoi")
                            .map_err(|e| e.to_string())?;
                        Ok(Typed { value: i.into(), ty: Type::Int })
                    }
                    (Type::Float, Type::Float) | (Type::Int, Type::Int) => Ok(v),
                    (to, from) => Err(format!(
                        "cast não suportado: '{}' → '{}'",
                        type_name(from),
                        type_name(to)
                    )),
                }
            }
        }
    }

    // ------------------- binários (tipos já unificados) -----------------

    fn gen_binary(&mut self, op: BinOp, lhs: &Expr, rhs: &Expr) -> Result<Typed<'ctx>, String> {
        use BinOp::*;
        let l = self.gen_expr(lhs)?;
        let r = self.gen_expr(rhs)?;
        match op {
            And | Or => {
                let l = self.as_bool(&l, "operando esquerdo")?;
                let r = self.as_bool(&r, "operando direito")?;
                // Sem curto-circuito (F5) — avalia ambos.
                let v = match op {
                    And => self.builder.build_and(l, r, "andtmp"),
                    Or => self.builder.build_or(l, r, "ortmp"),
                    _ => unreachable!(),
                }
                .map_err(|e| e.to_string())?;
                Ok(Typed { value: v.into(), ty: Type::Bool })
            }
            Eq | Ne => self.gen_equality(op, &l, &r),
            Lt | Le | Gt | Ge => self.gen_relational(op, &l, &r),
            Add | Sub | Mul | Div => self.gen_arith(op, &l, &r),
        }
    }

    /// `==`/`!=`: float (fcmp), int (icmp), bool, string (ponteiro —
    /// internada, então igualdade é identidade), enum (i64).
    fn gen_equality(&mut self, op: BinOp, l: &Typed<'ctx>, r: &Typed<'ctx>) -> Result<Typed<'ctx>, String> {
        let pred = match op {
            BinOp::Eq => IntPredicate::EQ,
            BinOp::Ne => IntPredicate::NE,
            _ => unreachable!(),
        };
        let v = match &l.ty {
            Type::Float => {
                let l = self.as_float(l, "operando esquerdo")?;
                let r = self.as_float(r, "operando direito")?;
                let fp = if pred == IntPredicate::EQ { FloatPredicate::OEQ } else { FloatPredicate::ONE };
                self.builder.build_float_compare(fp, l, r, "cmptmp")
            }
            Type::Int => self.builder.build_int_compare(
                pred,
                self.as_int(l, "operando esquerdo")?,
                self.as_int(r, "operando direito")?,
                "cmptmp",
            ),
            Type::Bool => self.builder.build_int_compare(
                pred,
                l.value.into_int_value(),
                r.value.into_int_value(),
                "cmptmp",
            ),
            // string: internada — igualdade vira comparação de ponteiro.
            Type::Str => {
                let i64_ty = self.context.i64_type();
                let la = self
                    .builder
                    .build_ptr_to_int(l.value.into_pointer_value(), i64_ty, "straddr")
                    .map_err(|e| e.to_string())?;
                let ra = self
                    .builder
                    .build_ptr_to_int(r.value.into_pointer_value(), i64_ty, "straddr")
                    .map_err(|e| e.to_string())?;
                self.builder.build_int_compare(pred, la, ra, "cmptmp")
            }
            // enum: i64.
            Type::Named(_) => self.builder.build_int_compare(
                pred,
                l.value.into_int_value(),
                r.value.into_int_value(),
                "cmptmp",
            ),
            other => {
                return Err(format!(
                    "'{}' não suporta comparação (checker deveria ter pego)",
                    type_name(other)
                ))
            }
        }
        .map_err(|e| e.to_string())?;
        Ok(Typed { value: v.into(), ty: Type::Bool })
    }

    /// `< <= > >=`: int (icmp com sinal) ou float (fcmp).
    fn gen_relational(&mut self, op: BinOp, l: &Typed<'ctx>, r: &Typed<'ctx>) -> Result<Typed<'ctx>, String> {
        let v = match &l.ty {
            Type::Float => {
                let l = self.as_float(l, "operando esquerdo")?;
                let r = self.as_float(r, "operando direito")?;
                let fp = match op {
                    BinOp::Lt => FloatPredicate::OLT,
                    BinOp::Le => FloatPredicate::OLE,
                    BinOp::Gt => FloatPredicate::OGT,
                    BinOp::Ge => FloatPredicate::OGE,
                    _ => unreachable!(),
                };
                self.builder.build_float_compare(fp, l, r, "cmptmp")
            }
            Type::Int => {
                let l = self.as_int(l, "operando esquerdo")?;
                let r = self.as_int(r, "operando direito")?;
                let ip = match op {
                    BinOp::Lt => IntPredicate::SLT,
                    BinOp::Le => IntPredicate::SLE,
                    BinOp::Gt => IntPredicate::SGT,
                    BinOp::Ge => IntPredicate::SGE,
                    _ => unreachable!(),
                };
                self.builder.build_int_compare(ip, l, r, "cmptmp")
            }
            other => {
                return Err(format!(
                    "'{}' não suporta comparação relacional (checker deveria ter pego)",
                    type_name(other)
                ))
            }
        }
        .map_err(|e| e.to_string())?;
        Ok(Typed { value: v.into(), ty: Type::Bool })
    }

    /// `+ - * /`: int usa i64 (divisão sdiv trunca para zero, como Go);
    /// float usa f64. Tipos já unificados pelo checker.
    fn gen_arith(&mut self, op: BinOp, l: &Typed<'ctx>, r: &Typed<'ctx>) -> Result<Typed<'ctx>, String> {
        match &l.ty {
            Type::Float => {
                let l = self.as_float(l, "operando esquerdo")?;
                let r = self.as_float(r, "operando direito")?;
                let f = match op {
                    BinOp::Add => self.builder.build_float_add(l, r, "addtmp"),
                    BinOp::Sub => self.builder.build_float_sub(l, r, "subtmp"),
                    BinOp::Mul => self.builder.build_float_mul(l, r, "multmp"),
                    BinOp::Div => self.builder.build_float_div(l, r, "divtmp"),
                    _ => unreachable!(),
                }
                .map_err(|e| e.to_string())?;
                Ok(Typed { value: f.into(), ty: Type::Float })
            }
            Type::Int => {
                let l = self.as_int(l, "operando esquerdo")?;
                let r = self.as_int(r, "operando direito")?;
                // sdiv: trunca para zero (Go) — não é floor.
                let i = match op {
                    BinOp::Add => self.builder.build_int_add(l, r, "addtmp"),
                    BinOp::Sub => self.builder.build_int_sub(l, r, "subtmp"),
                    BinOp::Mul => self.builder.build_int_mul(l, r, "multmp"),
                    BinOp::Div => self.builder.build_int_signed_div(l, r, "divtmp"),
                    _ => unreachable!(),
                }
                .map_err(|e| e.to_string())?;
                Ok(Typed { value: i.into(), ty: Type::Int })
            }
            other => Err(format!(
                "'{}' não suporta aritmética (checker deveria ter pego)",
                type_name(other)
            )),
        }
    }

    /// Interna uma string literal: uma global constante por conteúdo
    /// (tabela por módulo). `==`/`!=` entre strings vira comparação de
    /// ponteiro — correta porque literais iguais compartilham a global.
    fn intern_string(&mut self, s: &str) -> Result<PointerValue<'ctx>, String> {
        if let Some(p) = self.strings.get(s) {
            return Ok(*p);
        }
        let g = self
            .builder
            .build_global_string_ptr(s, "str")
            .map_err(|e| e.to_string())?;
        let ptr = g.as_pointer_value();
        self.strings.insert(s.to_string(), ptr);
        Ok(ptr)
    }

    /// Resolve o alvo de uma chamada (método ou função de módulo)
    /// e avalia os argumentos. Retorna função, args e símbolo.
    fn resolve_call(
        &mut self,
        callee: &Expr,
        args: &[Expr],
    ) -> Result<(FunctionValue<'ctx>, Vec<BasicMetadataValueEnum<'ctx>>, String), String> {
        if let Expr::Member { obj, field, .. } = callee {
            // Método: a.b(args) → símbolo "A.b", obj vira arg 0 (por REFERÊNCIA).
            let (oty, optr) = self.gen_lvalue(obj)?.ok_or_else(|| {
                "método exige uma variável (receiver por referência) — atribua o valor a uma variável primeiro".to_string()
            })?;
            let ty_name = type_name(&oty);
            if !self.structs.contains_key(&ty_name) {
                return Err(format!(
                    "método exige receiver struct, encontrou '{ty_name}'"
                ));
            }
            let symbol = format!("{ty_name}.{field}");
            let function = self
                .module
                .get_function(&symbol)
                .ok_or_else(|| format!("método '{symbol}' não encontrado"))?;
            let mut arg_vals = vec![optr.into()];
            for a in args {
                let v = self.gen_expr(a)?;
                arg_vals.push(v.value.into());
            }
            Ok((function, arg_vals, symbol))
        } else {
            let Expr::Ident(fname, _) = callee else {
                return Err("callee de chamada inválido".into());
            };
            let function = self
                .module
                .get_function(fname)
                .ok_or_else(|| format!("função '{fname}' não encontrada"))?;
            let mut arg_vals: Vec<BasicMetadataValueEnum> = Vec::new();
            for a in args {
                let v = self.gen_expr(a)?;
                arg_vals.push(v.value.into());
            }
            Ok((function, arg_vals, fname.clone()))
        }
    }

    /// Completa uma chamada: emite o call e tipa o retorno pelo símbolo.
    fn finish_call(
        &self,
        function: FunctionValue<'ctx>,
        args: &[BasicMetadataValueEnum<'ctx>],
        symbol: &str,
    ) -> Result<Typed<'ctx>, String> {
        let call = self
            .builder
            .build_call(function, args, "calltmp")
            .map_err(|e| e.to_string())?;
        match call.try_as_basic_value().basic() {
            Some(v) => {
                let ret_ty = self
                    .ret_types
                    .get(symbol)
                    .and_then(|t| t.clone())
                    .unwrap_or_else(|| Type::Float);
                Ok(Typed { value: v, ty: ret_ty })
            }
            None => Err(format!(
                "função '{symbol}' retorna void, mas o valor está sendo usado"
            )),
        }
    }

    /// Extrai IntValue (i1) de um bool, com erro claro.
    fn as_bool(&self, t: &Typed<'ctx>, what: &str) -> Result<IntValue<'ctx>, String> {
        if t.ty != Type::Bool {
            return Err(format!(
                "'{what}' deve ser bool, encontrou '{}'",
                type_name(&t.ty)
            ));
        }
        Ok(t.value.into_int_value())
    }

    /// Avalia uma condição de if/for: bool (F4 — sem truthiness).
    fn gen_cond(&mut self, expr: &Expr) -> Result<IntValue<'ctx>, String> {
        let t = self.gen_expr(expr)?;
        if t.ty == Type::Bool {
            return Ok(t.value.into_int_value());
        }
        Err(format!(
            "condição deve ser bool, encontrou '{}'",
            type_name(&t.ty)
        ))
    }

    // ======================== execução JIT ============================

    /// Roda o módulo via JIT e executa `main` (float, int ou void).
    pub fn run_main(&self) -> Result<MainResult, String> {
        let engine = self
            .module
            .create_jit_execution_engine(OptimizationLevel::None)
            .map_err(|e| e.to_string())?;
        let _main = self
            .module
            .get_function("main")
            .ok_or("função 'main' não encontrada no programa")?;

        match self.ret_types.get("main").cloned().flatten() {
            Some(Type::Float) => {
                type MainF = unsafe extern "C" fn() -> f64;
                // SAFETY: main foi gerada por nós com assinatura compatível.
                let f: JitFunction<MainF> =
                    unsafe { engine.get_function("main") }.map_err(|e| e.to_string())?;
                unsafe { Ok(MainResult::Float(f.call())) }
            }
            Some(Type::Int) => {
                type MainI = unsafe extern "C" fn() -> i64;
                // SAFETY: idem.
                let f: JitFunction<MainI> =
                    unsafe { engine.get_function("main") }.map_err(|e| e.to_string())?;
                unsafe { Ok(MainResult::Int(f.call())) }
            }
            None => {
                type MainV = unsafe extern "C" fn();
                let f: JitFunction<MainV> =
                    unsafe { engine.get_function("main") }.map_err(|e| e.to_string())?;
                unsafe {
                    f.call();
                }
                Ok(MainResult::Void)
            }
            Some(other) => Err(format!(
                "main deve retornar float, int ou void, encontrou '{}'",
                type_name(&other)
            )),
        }
    }

    // =========================== F1: expressão ========================

    /// Compila uma expressão como corpo de `top() -> f64` (caminho F1).
    /// int é promovido para float na saída (sitofp); string/bool/struct
    /// não têm representação em `top`.
    pub fn compile_top(&mut self, expr: &Expr) -> Result<(), String> {
        let f64_ty = self.context.f64_type();
        let fn_ty = f64_ty.fn_type(&[], false);
        let function = self.module.add_function("top", fn_ty, None);
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);
        let value = self.gen_expr(expr)?;
        let value = match value.ty {
            Type::Float => self.as_float(&value, "expressão")?,
            Type::Int => self
                .builder
                .build_signed_int_to_float(
                    self.as_int(&value, "expressão")?,
                    f64_ty,
                    "itof",
                )
                .map_err(|e| e.to_string())?,
            other => {
                return Err(format!(
                    "expressão deve ser int ou float, encontrou '{}'",
                    type_name(&other)
                ))
            }
        };
        self.builder.build_return(Some(&value)).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Executa `top()` (caminho F1).
    pub fn run_jit(&self) -> Result<f64, String> {
        let engine = self
            .module
            .create_jit_execution_engine(OptimizationLevel::None)
            .map_err(|e| e.to_string())?;
        // SAFETY: top foi gerada por nós com assinatura extern "C" fn() -> f64.
        let top: JitFunction<TopFn> =
            unsafe { engine.get_function("top") }.map_err(|e| e.to_string())?;
        unsafe { Ok(top.call()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{parse_expr, parse_program};
    use inkwell::context::Context;

    // ---- caminho F1 (expressão pura) ----

    fn eval(src: &str) -> f64 {
        let context = Context::create();
        let mut cg = Codegen::new(&context);
        let expr = parse_expr(src).unwrap();
        let expr = crate::checker::check_expr(&expr).unwrap();
        cg.compile_top(&expr).unwrap();
        cg.run_jit().unwrap()
    }

    #[test]
    fn jit_executes_arithmetic() {
        assert_eq!(eval("1.0 + 2.0"), 3.0);
        assert_eq!(eval("1 + 2"), 3.0); // int promovido na saída
        assert_eq!(eval("1.0 + 2.0 * 3.0"), 7.0);
        assert_eq!(eval("(1 + 2) * 3"), 9.0);
        assert_eq!(eval("10.0 / 4.0"), 2.5);
        assert_eq!(eval("10 / 4"), 2.0); // divisão de int trunca
        assert_eq!(eval("1 + 2.5"), 3.5); // int promovido no binário
    }

    // ---- caminho F2b (programa com main) ----

    fn eval_program(src: &str) -> Result<Option<f64>, String> {
        let program = parse_program(src).map_err(|e| e.to_string())?;
        let program = crate::checker::check_program(&program).map_err(|e| e.to_string())?;
        let context = Context::create();
        let mut cg = Codegen::new(&context);
        cg.compile_program(&program)?;
        match cg.run_main()? {
            MainResult::Float(f) => Ok(Some(f)),
            MainResult::Int(i) => Ok(Some(i as f64)),
            MainResult::Void => Ok(None),
        }
    }

    fn eval_main(src: &str) -> f64 {
        eval_program(src).unwrap().unwrap()
    }

    #[test]
    fn functions_and_calls() {
        let src = r#"
            func add(a: float, b: float): float {
                return a + b;
            }
            func square(x: float): float {
                return x * x;
            }
            func main(): float {
                return square(add(2.0, 3.0));
            }
        "#;
        assert_eq!(eval_main(src), 25.0);
    }

    #[test]
    fn locals_with_let() {
        let src = r#"
            func main(): float {
                let a = 2.0;
                let b = a * 3.0;
                let c = b + 1.0;
                return c;
            }
        "#;
        assert_eq!(eval_main(src), 7.0);
    }

    #[test]
    fn forward_reference() {
        let src = r#"
            func main(): float {
                return f(4.0);
            }
            func f(x: float): float {
                return g(x) + 1.0;
            }
            func g(x: float): float {
                return x * 10.0;
            }
        "#;
        assert_eq!(eval_main(src), 41.0);
    }

    #[test]
    fn void_main_executes() {
        let src = r#"
            func main() {
                let x = 1.0;
                return;
            }
        "#;
        assert!(eval_program(src).unwrap().is_none());
    }

    #[test]
    fn missing_main_is_error() {
        let src = "func f(): float { return 1.0; }";
        assert!(eval_program(src).is_err());
    }

    #[test]
    fn undefined_variable_is_error() {
        let src = "func main(): float { return x; }";
        assert!(eval_program(src).unwrap_err().contains("x"));
    }

    #[test]
    fn unknown_type_is_error() {
        let src = "func main(): banana { return 1.0; }";
        assert!(eval_program(src).unwrap_err().contains("banana"));
    }

    // ---- caminho F2c (structs, receiver, métodos) ----

    #[test]
    fn struct_literal_and_method() {
        let src = r#"
            struct Point {
                x: float;
                y: float;
            }
            func (p: Point) len_sq(): float {
                return p.x * p.x + p.y * p.y;
            }
            func make(x: float, y: float): Point {
                return Point { x: x, y: y };
            }
            func main(): float {
                let p = make(3.0, 4.0);
                return p.len_sq();
            }
        "#;
        assert_eq!(eval_main(src), 25.0);
    }

    #[test]
    fn member_of_literal_directly() {
        let src = r#"
            struct Point {
                x: float;
                y: float;
            }
            func main(): float {
                return (Point { x: 2.0, y: 5.0 }).x * 3.0;
            }
        "#;
        assert_eq!(eval_main(src), 6.0);
    }

    #[test]
    fn struct_passed_by_value() {
        let src = r#"
            struct Point {
                x: float;
                y: float;
            }
            func sum(p: Point): float {
                return p.x + p.y;
            }
            func main(): float {
                return sum(Point { x: 10.0, y: 20.0 });
            }
        "#;
        assert_eq!(eval_main(src), 30.0);
    }

    #[test]
    fn method_with_extra_args() {
        let src = r#"
            struct Point {
                x: float;
                y: float;
            }
            func (p: Point) dist(q: Point): float {
                let dx = p.x - q.x;
                let dy = p.y - q.y;
                return dx * dx + dy * dy;
            }
            func main(): float {
                let a = Point { x: 0.0, y: 0.0 };
                let b = Point { x: 3.0, y: 4.0 };
                return a.dist(b);
            }
        "#;
        assert_eq!(eval_main(src), 25.0);
    }

    #[test]
    fn member_on_non_struct_is_error() {
        let src = "func main(): float { let x = 1.0; return x.field; }";
        assert!(eval_program(src).unwrap_err().contains("struct"));
    }

    #[test]
    fn unknown_method_is_error() {
        let src = r#"
            struct A { v: float; }
            func main(): float {
                let a = A { v: 1.0 };
                return a.nope();
            }
        "#;
        assert!(eval_program(src).unwrap_err().contains("nope"));
    }

    #[test]
    fn missing_struct_field_is_error() {
        let src = r#"
            struct A { x: float; y: float; }
            func main(): float {
                let a = A { x: 1.0 };
                return a.x;
            }
        "#;
        assert!(eval_program(src).unwrap_err().contains("y"));
    }

    #[test]
    fn nested_struct_read_and_write() {
        let src = r#"
            struct Vec2 {
                x: float;
                y: float;
            }
            struct Citizen {
                home: Vec2;
                pos: Vec2;
            }
            func main(): float {
                let c = Citizen { home: Vec2 { x: 10.0, y: 20.0 }, pos: Vec2 { x: 0.0, y: 0.0 } };
                c.pos.x = c.home.x * 0.5;
                c.pos.y = c.home.y * 0.5;
                return c.pos.x + c.pos.y;
            }
        "#;
        assert_eq!(eval_main(src), 15.0);
    }

    #[test]
    fn field_assignment() {
        let src = r#"
            struct Point { x: float; y: float; }
            func main(): float {
                let p = Point { x: 1.0, y: 2.0 };
                p.x = 10.0;
                return p.x + p.y;
            }
        "#;
        assert_eq!(eval_main(src), 12.0);
    }

    #[test]
    fn enum_as_value_and_assignment() {
        let src = r#"
            enum Mood { Happy, Neutral, Stressed }
            struct Citizen { mood: Mood; }
            func (c: Citizen) set_mood(m: Mood) {
                c.mood = m;
            }
            func main(): float {
                let c = Citizen { mood: Mood.Happy };
                c.set_mood(Mood.Stressed);
                c.mood = Mood.Neutral;
                if c.mood == Mood.Neutral {
                    return 1.0;
                }
                return 0.0;
            }
        "#;
        assert_eq!(eval_main(src), 1.0);
    }

    #[test]
    fn assign_to_temporary_is_error() {
        let src = r#"
            struct Point { x: float; y: float; }
            func main(): float {
                (Point { x: 1.0, y: 2.0 }).x = 5.0;
                return 1.0;
            }
        "#;
        assert!(eval_program(src).unwrap_err().contains("alvo"));
    }

    #[test]
    fn recursive_struct_is_error() {
        let src = r#"
            struct A { b: B; }
            struct B { a: A; }
            func main(): float { return 1.0; }
        "#;
        let err = eval_program(src).unwrap_err();
        assert!(err.contains("recursiva"));
    }

    #[test]
    fn unknown_enum_variant_is_error() {
        let src = r#"
            enum Mood { Happy }
            func main(): float { return Mood.Sad; }
        "#;
        assert!(eval_program(src).unwrap_err().contains("Sad"));
    }

    // ---- F3: controle de fluxo ----

    #[test]
    fn if_else_basic() {
        let src = r#"
            func main(): float {
                let x = 5.0;
                if x > 3.0 {
                    return 10.0;
                } else {
                    return 20.0;
                }
            }
        "#;
        assert_eq!(eval_main(src), 10.0);
    }

    #[test]
    fn if_without_else_and_else_if() {
        let src = r#"
            func main(): float {
                let x = 2.0;
                if x > 5.0 {
                    return 1.0;
                } else if x > 1.0 {
                    return 2.0;
                } else if x > 0.0 {
                    return 3.0;
                } else {
                    return 4.0;
                }
            }
        "#;
        assert_eq!(eval_main(src), 2.0);
    }

    #[test]
    fn if_no_else_falls_through() {
        let src = r#"
            func main(): float {
                let x = 0.0;
                if x > 3.0 {
                    x = 99.0;
                }
                return x;
            }
        "#;
        assert_eq!(eval_main(src), 0.0);
    }

    #[test]
    fn comparisons_and_logic() {
        let src = r#"
            func main(): float {
                let a = 1.0;
                let b = 2.0;
                if a > b && b <= 2.0 && a != b {
                    return 1.0;
                }
                if !(a > b) && (a >= 1.0 || b < 1.0) {
                    return 2.0;
                }
                return 0.0;
            }
        "#;
        assert_eq!(eval_main(src), 2.0);
    }

    #[test]
    fn bool_variable_as_condition() {
        let src = r#"
            func main(): float {
                let flag = 3.0 < 5.0;
                if flag {
                    return 42.0;
                }
                return 0.0;
            }
        "#;
        assert_eq!(eval_main(src), 42.0);
    }

    #[test]
    fn for_counter_loop() {
        let src = r#"
            func main(): float {
                let sum = 0.0;
                for let i = 0.0; i < 5.0; i = i + 1.0 {
                    sum = sum + i;
                }
                return sum;
            }
        "#;
        assert_eq!(eval_main(src), 10.0); // 0+1+2+3+4
    }

    #[test]
    fn for_condition_style() {
        let src = r#"
            func main(): float {
                let i = 0.0;
                for i < 4.0 {
                    i = i + 2.0;
                }
                return i;
            }
        "#;
        assert_eq!(eval_main(src), 4.0);
    }

    #[test]
    fn for_infinite_with_break() {
        let src = r#"
            func main(): float {
                let i = 0.0;
                for {
                    i = i + 1.0;
                    if i >= 7.0 {
                        break;
                    }
                }
                return i;
            }
        "#;
        assert_eq!(eval_main(src), 7.0);
    }

    #[test]
    fn continue_skips_iteration() {
        let src = r#"
            func main(): float {
                let sum = 0.0;
                for let i = 0.0; i < 5.0; i = i + 1.0 {
                    if i == 2.0 {
                        continue;
                    }
                    sum = sum + i;
                }
                return sum;
            }
        "#;
        assert_eq!(eval_main(src), 8.0); // 0+1+3+4
    }

    #[test]
    fn nested_loops_with_break() {
        let src = r#"
            func main(): float {
                let count = 0.0;
                for let i = 0.0; i < 3.0; i = i + 1.0 {
                    for let j = 0.0; j < 3.0; j = j + 1.0 {
                        if j >= 2.0 {
                            break;
                        }
                        count = count + 1.0;
                    }
                }
                return count;
            }
        "#;
        assert_eq!(eval_main(src), 6.0); // 2 por iteração externa
    }

    #[test]
    fn unary_minus_and_negative_numbers() {
        let src = r#"
            func main(): float {
                let x = -3.0;
                return -x * 2.0;
            }
        "#;
        assert_eq!(eval_main(src), 6.0);
    }

    #[test]
    fn break_outside_loop_is_error() {
        let src = "func main(): float { break; }";
        assert!(eval_program(src).unwrap_err().contains("loop"));
    }

    #[test]
    fn struct_condition_is_error() {
        let src = r#"
            struct P { x: float; }
            func main(): float {
                let p = P { x: 1.0 };
                if p {
                    return 1.0;
                }
                return 0.0;
            }
        "#;
        assert!(eval_program(src).unwrap_err().contains("condição"));
    }

    #[test]
    fn simulation_population() {
        let src = r#"
            enum Mood { Happy, Neutral, Stressed }
            struct Citizen {
                home: float;
                mood: Mood;
            }
            func make_population(n: float): float {
                let count = 0.0;
                for let i = 0.0; i < n; i = i + 1.0 {
                    let c = Citizen { home: i, mood: Mood.Happy };
                    if c.home > n / 2.0 {
                        c.mood = Mood.Stressed;
                    }
                    if c.mood == Mood.Stressed {
                        count = count + 1.0;
                    }
                }
                return count;
            }
            func main(): float {
                return make_population(10.0);
            }
        "#;
        // 10 cidadãos; home = i (0..9); stressed = home > 5.0 → 6,7,8,9 = 4
        assert_eq!(eval_main(src), 4.0);
    }

    #[test]
    fn citizen_end_to_end() {
        let src = r#"
            enum Mood { Happy, Neutral, Stressed }
            struct Vec2 { x: float; y: float; }
            struct Citizen {
                home: Vec2;
                work: Vec2;
                pos: Vec2;
                mood: Mood;
            }
            func (c: Citizen) go_home(dt: float) {
                c.pos.x = c.pos.x + (c.home.x - c.pos.x) * dt;
                c.pos.y = c.pos.y + (c.home.y - c.pos.y) * dt;
            }
            func (c: Citizen) set_mood(m: Mood) {
                c.mood = m;
            }
            func main(): float {
                let c = Citizen {
                    home: Vec2 { x: 100.0, y: 0.0 },
                    work: Vec2 { x: 0.0, y: 100.0 },
                    pos: Vec2 { x: 0.0, y: 0.0 },
                    mood: Mood.Happy,
                };
                c.go_home(0.5);
                c.set_mood(Mood.Stressed);
                return c.pos.x + c.pos.y;
            }
        "#;
        // pos.x = 0 + (100-0)*0.5 = 50; pos.y = 0
        assert_eq!(eval_main(src), 50.0);
    }

    // ---- F4: int/float, bool, strings, casts ----

    #[test]
    fn int_arithmetic_and_division_truncates() {
        let src = "func main(): int { return 1 + 2 * 3; }";
        assert_eq!(eval_main(src), 7.0);
        let src = "func main(): int { return 7 / 2; }";
        assert_eq!(eval_main(src), 3.0); // sdiv trunca
        let src = "func main(): int { return -7 / 2; }";
        assert_eq!(eval_main(src), -3.0); // trunca para zero (Go)
        let src = "func main(): int { return -3 + 10; }";
        assert_eq!(eval_main(src), 7.0);
    }

    #[test]
    fn int_annotated_let_and_int_comparisons() {
        let src = r#"
            func main(): int {
                let x: int = 5;
                let y = 3;
                if x > y && y >= 3 && x != y {
                    return x - y;
                }
                return 0;
            }
        "#;
        assert_eq!(eval_main(src), 2.0);
    }

    #[test]
    fn implicit_int_to_float_coercion() {
        // chamada, let anotado e binário misto
        let src = r#"
            func add(a: float, b: float): float { return a + b; }
            func main(): float {
                let x: float = 1;
                let y = 2.5 + 1;
                return add(1, 2) + x + y;
            }
        "#;
        assert_eq!(eval_main(src), 7.5); // 3.0 + 1.0 + 3.5
    }

    #[test]
    fn explicit_cast_float_to_int_truncates() {
        let src = "func main(): int { return int(3.9); }";
        assert_eq!(eval_main(src), 3.0);
        let src = "func main(): int { return int(-3.9); }";
        assert_eq!(eval_main(src), -3.0);
        let src = "func main(): float { return float(3); }";
        assert_eq!(eval_main(src), 3.0);
    }

    #[test]
    fn float_to_int_without_cast_is_error() {
        let src = "func main(): int { let x = 1.5; return x; }";
        assert!(eval_program(src).unwrap_err().contains("cast"));
        let src = "func main(): float { let x: int = 1.5; return 1.0; }";
        assert!(eval_program(src).unwrap_err().contains("int(x)"));
    }

    #[test]
    fn bool_literals_and_logic() {
        let src = r#"
            func main(): float {
                let flag = true;
                if flag && !false && (3 > 2) {
                    return 1.0;
                }
                return 0.0;
            }
        "#;
        assert_eq!(eval_main(src), 1.0);
    }

    #[test]
    fn bool_equality() {
        let src = r#"
            func main(): float {
                if true == true && false != true {
                    return 1.0;
                }
                return 0.0;
            }
        "#;
        assert_eq!(eval_main(src), 1.0);
    }

    #[test]
    fn mixed_int_float_comparison() {
        let src = r#"
            func main(): float {
                if 1 < 2.5 {
                    return 1.0;
                }
                return 0.0;
            }
        "#;
        assert_eq!(eval_main(src), 1.0);
    }

    #[test]
    fn string_equality_uses_interning() {
        let src = r#"
            func main(): float {
                let a = "hello";
                let b = "hello";
                let c = "world";
                if a == b && a != c && "hello" == a {
                    return 1.0;
                }
                return 0.0;
            }
        "#;
        assert_eq!(eval_main(src), 1.0);
    }

    #[test]
    fn strings_in_params_and_returns() {
        let src = r#"
            func greet(name: string): string {
                return name;
            }
            func main(): float {
                let s = greet("forge");
                if s == "forge" && greet("a") != greet("b") {
                    return 1.0;
                }
                return 0.0;
            }
        "#;
        assert_eq!(eval_main(src), 1.0);
    }

    #[test]
    fn string_in_struct_and_condition_must_be_bool() {
        let src = r#"
            struct Person {
                name: string;
                age: int;
            }
            func main(): float {
                let p = Person { name: "ada", age: 36 };
                if p.name == "ada" && p.age >= 30 {
                    return 1.0;
                }
                return 0.0;
            }
        "#;
        assert_eq!(eval_main(src), 1.0);
        // condição não-bool agora é erro de tipo
        let src = "func main(): float { let x = 1.0; if x { return 1.0; } return 0.0; }";
        assert!(eval_program(src).unwrap_err().contains("bool"));
    }

    #[test]
    fn int_main_executes() {
        let src = "func main(): int { return 42; }";
        let result = eval_program(src).unwrap();
        assert_eq!(result, Some(42.0));
    }

    #[test]
    fn enum_as_int_and_comparison() {
        let src = r#"
            enum Level { Low, Mid, High }
            func main(): int {
                let l = Level.High;
                if l == Level.High {
                    return 1;
                }
                return 0;
            }
        "#;
        assert_eq!(eval_main(src), 1.0);
    }

    #[test]
    fn checker_rejects_bad_programs() {
        // string + string
        let src = r#"func main(): float { let s = "a" + "b"; return 1.0; }"#;
        assert!(eval_program(src).is_err());
        // return sem valor em função não-void
        let src = "func main(): float { return; }";
        assert!(eval_program(src).is_err());
        // void com return de valor
        let src = "func main() { return 1.0; }";
        assert!(eval_program(src).is_err());
        // aridade errada
        let src = "func f(a: float) { } func main() { f(1.0, 2.0); }";
        assert!(eval_program(src).is_err());
        // variável fora do escopo do bloco
        let src = "func main(): float { if true { let x = 1.0; } return x; }";
        assert!(eval_program(src).is_err());
    }
}
