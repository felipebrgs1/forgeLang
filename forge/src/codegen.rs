//! Codegen: AST → LLVM IR, e execução via JIT.
//!
//! F1:  expressão única → função `top()` → JIT
//! F2b: programa inteiro → funções, variáveis, chamadas, `main`
//! F2c: structs (LLVM agregados + GEP), receiver como param 0,
//!      member access (`p.x`) e chamadas de método (`p.len_sq()`)
//!
//! Um mini type-check vive aqui (tipo de cada expressão) até o type
//! checker real da F4. Tipos hoje: `float` e structs de campos float.

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
use inkwell::{FloatPredicate, OptimizationLevel};
use std::collections::HashMap;

/// Assinatura da função top-level do caminho F1 (expressão).
pub type TopFn = unsafe extern "C" fn() -> f64;

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
    /// Internamente são floats (F4 troca por tipos reais).
    enums: HashMap<String, HashMap<String, f64>>,
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
            current_ret: None,
            current_fn: None,
            block_open: true,
            loop_stack: Vec::new(),
        }
    }

    // ======================= tipos e helpers =========================

    fn type_name(ty: &Type) -> String {
        match ty {
            Type::Named(name) => name.clone(),
        }
    }

    fn is_float(ty: &Type) -> bool {
        matches!(ty, Type::Named(n) if n == "float")
    }

    /// Tipo LLVM correspondente a um tipo da linguagem.
    fn llvm_type(&self, ty: &Type) -> Result<BasicTypeEnum<'ctx>, String> {
        match ty {
            // enums são floats internamente até a F4
            Type::Named(n) if n == "float" || self.enums.contains_key(n) => {
                Ok(self.context.f64_type().into())
            }
            Type::Named(n) if n == "bool" => Ok(self.context.bool_type().into()),
            Type::Named(n) => self
                .structs
                .get(n)
                .map(|(st, _)| (*st).into())
                .ok_or_else(|| format!("tipo '{n}' desconhecido (struct não declarada ou tipo não suportado)")),
        }
    }

    /// Extrai FloatValue, com erro claro se o tipo não é numérico
    /// (float ou enum — enums são f64 internamente até a F4).
    fn as_float(&self, t: &Typed<'ctx>, what: &str) -> Result<FloatValue<'ctx>, String> {
        let tn = Self::type_name(&t.ty);
        if !(Self::is_float(&t.ty) || self.enums.contains_key(&tn)) {
            return Err(format!("'{what}' deve ser numérico, encontrou '{tn}'"));
        }
        Ok(t.value.into_float_value())
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

    /// Passo 0: enums viram mapas de constantes (valores 0, 1, 2...).
    fn declare_enums(&mut self, program: &Program) -> Result<(), String> {
        for decl in &program.decls {
            if let Decl::Enum(e) = decl {
                let mut variants = HashMap::new();
                for (i, v) in e.variants.iter().enumerate() {
                    variants.insert(v.clone(), i as f64);
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
        // 2. Valida: cada campo é float | enum | struct declarada.
        for (name, fields) in &raw {
            for (fname, ty) in fields {
                let tn = Self::type_name(ty);
                if !(Self::is_float(ty) || self.enums.contains_key(&tn) || raw.contains_key(&tn)) {
                    return Err(format!(
                        "struct '{name}': campo '{fname}' de tipo '{tn}' desconhecido"
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
            let tn = Self::type_name(ty);
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
            Some(r) => format!("{}.{}", Self::type_name(&r.ty), f.name),
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

        // Retorno implícito: 0.0 para float, void caso contrário.
        if self.block_open {
            match &f.ret {
                Some(t) if Self::is_float(t) => {
                    let zero = self.context.f64_type().const_float(0.0);
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
                    if Self::type_name(annot) != Self::type_name(&val.ty) {
                        return Err(format!(
                            "tipo anotado '{0}' não bate com o valor '{1}' (inferência real: F4)",
                            Self::type_name(annot),
                            Self::type_name(&val.ty)
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
                            if Self::type_name(expected) != Self::type_name(&val.ty) {
                                return Err(format!(
                                    "retorno '{}' não bate com o tipo declarado '{}'",
                                    Self::type_name(&val.ty),
                                    Self::type_name(expected)
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
                        if Self::type_name(&tty) != Self::type_name(&val.ty) {
                            return Err(format!(
                                "atribuição: alvo '{}' não aceita valor '{}'",
                                Self::type_name(&tty),
                                Self::type_name(&val.ty)
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
                let ty_name = Self::type_name(&oty);
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

    fn gen_expr(&self, expr: &Expr) -> Result<Typed<'ctx>, String> {
        match expr {
            Expr::Number(n, _) => Ok(Typed {
                value: self.context.f64_type().const_float(*n).into(),
                ty: Type::Named("float".into()),
            }),
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
            Expr::Binary { op, lhs, rhs, .. } => {
                use BinOp::*;
                // Comparações e lógicos
                let cmp = match op {
                    Eq | Ne | Lt | Le | Gt | Ge => Some(*op),
                    _ => None,
                };
                if let Some(cmp) = cmp {
                    let l = self.gen_expr(lhs)?;
                    let r = self.gen_expr(rhs)?;
                    let l = self.as_float(&l, "operando esquerdo")?;
                    let r = self.as_float(&r, "operando direito")?;
                    let pred = match cmp {
                        Eq => FloatPredicate::OEQ,
                        Ne => FloatPredicate::ONE,
                        Lt => FloatPredicate::OLT,
                        Le => FloatPredicate::OLE,
                        Gt => FloatPredicate::OGT,
                        Ge => FloatPredicate::OGE,
                        _ => unreachable!(),
                    };
                    let v = self
                        .builder
                        .build_float_compare(pred, l, r, "cmptmp")
                        .map_err(|e| e.to_string())?;
                    return Ok(Typed {
                        value: v.into(),
                        ty: Type::Named("bool".into()),
                    });
                }
                match op {
                    And | Or => {
                        let l = self.gen_expr(lhs)?;
                        let r = self.gen_expr(rhs)?;
                        let l = self.as_bool(&l, "operando esquerdo")?;
                        let r = self.as_bool(&r, "operando direito")?;
                        // Sem curto-circuito (F5) — avalia ambos.
                        let v = match op {
                            And => self.builder.build_and(l, r, "andtmp"),
                            Or => self.builder.build_or(l, r, "ortmp"),
                            _ => unreachable!(),
                        }
                        .map_err(|e| e.to_string())?;
                        Ok(Typed {
                            value: v.into(),
                            ty: Type::Named("bool".into()),
                        })
                    }
                    _ => {
                        let l = self.gen_expr(lhs)?;
                        let r = self.gen_expr(rhs)?;
                        let l = self.as_float(&l, "operando esquerdo")?;
                        let r = self.as_float(&r, "operando direito")?;
                        let result = match op {
                            BinOp::Add => self.builder.build_float_add(l, r, "addtmp"),
                            BinOp::Sub => self.builder.build_float_sub(l, r, "subtmp"),
                            BinOp::Mul => self.builder.build_float_mul(l, r, "multmp"),
                            BinOp::Div => self.builder.build_float_div(l, r, "divtmp"),
                            _ => unreachable!(),
                        }
                        .map_err(|e| e.to_string())?;
                        Ok(Typed {
                            value: result.into(),
                            ty: Type::Named("float".into()),
                        })
                    }
                }
            }
            Expr::StructLit { name, fields, .. } => {
                let (st, def) = self
                    .structs
                    .get(name)
                    .ok_or_else(|| format!("struct '{name}' não declarada"))?;
                // Materializa num alloca temporário e carrega como valor.
                let temp = self
                    .builder
                    .build_alloca(*st, name)
                    .map_err(|e| e.to_string())?;
                let mut missing: Vec<&str> = def.fields.iter().map(|(n, _)| n.as_str()).collect();
                for (fname, fexpr) in fields {
                    let idx = def.index(fname).ok_or_else(|| {
                        format!("struct '{name}' não tem campo '{fname}'")
                    })?;
                    let v = self.gen_expr(fexpr)?;
                    let ptr = self
                        .builder
                        .build_struct_gep(*st, temp, idx, "gep")
                        .map_err(|e| e.to_string())?;
                    self.builder.build_store(ptr, v.value).map_err(|e| e.to_string())?;
                    missing.retain(|n| *n != fname);
                }
                if let Some(fname) = missing.first() {
                    return Err(format!("struct literal '{name}' sem o campo '{fname}'"));
                }
                let val = self
                    .builder
                    .build_load(*st, temp, "structval")
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
                            value: self.context.f64_type().const_float(*value).into(),
                            ty: Type::Named(enum_name.clone()),
                        });
                    }
                }
                // Struct member: se o obj é lvalue, GEP direto no ponteiro
                // (mais eficiente que materializar temp).
                if let Some((oty, optr)) = self.gen_lvalue(obj)? {
                    let ty_name = Self::type_name(&oty);
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
                    let ty_name = Self::type_name(&obj_v.ty);
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
                    let v = self.as_float(&v, "operando")?;
                    let r = self
                        .builder
                        .build_float_neg(v, "negtmp")
                        .map_err(|e| e.to_string())?;
                    Ok(Typed {
                        value: r.into(),
                        ty: Type::Named("float".into()),
                    })
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
                        ty: Type::Named("bool".into()),
                    })
                }
            },
            Expr::Str(..) => {
                Err("strings ainda não suportadas no codegen (F4)".into())
            }
        }
    }

    /// Resolve o alvo de uma chamada (método ou função de módulo)
    /// e avalia os argumentos. Retorna função, args e símbolo.
    fn resolve_call(
        &self,
        callee: &Expr,
        args: &[Expr],
    ) -> Result<(FunctionValue<'ctx>, Vec<BasicMetadataValueEnum<'ctx>>, String), String> {
        if let Expr::Member { obj, field, .. } = callee {
            // Método: a.b(args) → símbolo "A.b", obj vira arg 0 (por REFERÊNCIA).
            let (oty, optr) = self.gen_lvalue(obj)?.ok_or_else(|| {
                "método exige uma variável (receiver por referência) — atribua o valor a uma variável primeiro".to_string()
            })?;
            let ty_name = Self::type_name(&oty);
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
                    .unwrap_or_else(|| Type::Named("float".into()));
                Ok(Typed { value: v, ty: ret_ty })
            }
            None => Err(format!(
                "função '{symbol}' retorna void, mas o valor está sendo usado"
            )),
        }
    }

    /// Extrai IntValue (i1) de um bool, com erro claro.
    fn as_bool(&self, t: &Typed<'ctx>, what: &str) -> Result<IntValue<'ctx>, String> {
        if !matches!(t.ty, Type::Named(ref n) if n == "bool") {
            return Err(format!(
                "'{what}' deve ser bool, encontrou '{}'",
                Self::type_name(&t.ty)
            ));
        }
        Ok(t.value.into_int_value())
    }

    /// Avalia uma condição de if/for: bool direto, ou numérico != 0.
    fn gen_cond(&self, expr: &Expr) -> Result<IntValue<'ctx>, String> {
        let t = self.gen_expr(expr)?;
        let tn = Self::type_name(&t.ty);
        if tn == "bool" {
            return Ok(t.value.into_int_value());
        }
        if Self::is_float(&t.ty) || self.enums.contains_key(&tn) {
            let v = t.value.into_float_value();
            let zero = self.context.f64_type().const_float(0.0);
            return self
                .builder
                .build_float_compare(FloatPredicate::ONE, v, zero, "cond")
                .map_err(|e| e.to_string());
        }
        Err(format!("condição deve ser bool ou numérica, encontrou '{tn}'"))
    }

    // ======================== execução JIT ============================

    /// Roda o módulo via JIT e executa `main`, retornando o valor se float.
    pub fn run_main(&self) -> Result<Option<f64>, String> {
        let engine = self
            .module
            .create_jit_execution_engine(OptimizationLevel::None)
            .map_err(|e| e.to_string())?;
        let _main = self
            .module
            .get_function("main")
            .ok_or("função 'main' não encontrada no programa")?;

        match self.ret_types.get("main").cloned().flatten() {
            Some(Type::Named(n)) if n == "float" => {
                type MainF = unsafe extern "C" fn() -> f64;
                // SAFETY: main foi gerada por nós com assinatura compatível.
                let f: JitFunction<MainF> =
                    unsafe { engine.get_function("main") }.map_err(|e| e.to_string())?;
                unsafe { Ok(Some(f.call())) }
            }
            None => {
                type MainV = unsafe extern "C" fn();
                let f: JitFunction<MainV> =
                    unsafe { engine.get_function("main") }.map_err(|e| e.to_string())?;
                unsafe {
                    f.call();
                }
                Ok(None)
            }
            Some(other) => Err(format!(
                "main deve retornar float ou void, encontrou '{}'",
                Self::type_name(&other)
            )),
        }
    }

    // =========================== F1: expressão ========================

    /// Compila uma expressão como corpo de `top() -> f64` (caminho F1).
    pub fn compile_top(&mut self, expr: &Expr) -> Result<(), String> {
        let f64_ty = self.context.f64_type();
        let fn_ty = f64_ty.fn_type(&[], false);
        let function = self.module.add_function("top", fn_ty, None);
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);
        let value = self.gen_expr(expr)?;
        let value = self.as_float(&value, "expressão")?;
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
        cg.compile_top(&expr).unwrap();
        cg.run_jit().unwrap()
    }

    #[test]
    fn jit_executes_arithmetic() {
        assert_eq!(eval("1 + 2"), 3.0);
        assert_eq!(eval("1 + 2 * 3"), 7.0);
        assert_eq!(eval("(1 + 2) * 3"), 9.0);
        assert_eq!(eval("10 / 4"), 2.5);
    }

    // ---- caminho F2b (programa com main) ----

    fn eval_program(src: &str) -> Result<Option<f64>, String> {
        let program = parse_program(src).map_err(|e| e.to_string())?;
        let context = Context::create();
        let mut cg = Codegen::new(&context);
        cg.compile_program(&program)?;
        cg.run_main()
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
        let src = "func main(): int { return 1; }";
        assert!(eval_program(src).unwrap_err().contains("int"));
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
                return c.mood + 1.0;
            }
        "#;
        // Neutral = 1.0, então 1.0 + 1.0 = 2.0
        assert_eq!(eval_main(src), 2.0);
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
}
