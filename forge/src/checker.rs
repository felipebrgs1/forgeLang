//! Type checker (F4): valida o programa e produz a versão "tipada".
//!
//! O AST verificado é o mesmo AST, com `Cast` inseridos onde a coerção
//! implícita int→float acontece. Regras (estilo Go):
//!
//!   - primitivos: int, float, bool, string, void (só em retorno)
//!   - `true`/`false` são bool; `42` é int; `42.0` é float
//!   - divisão de int trunca (codegen usa sdiv)
//!   - coerção assimétrica: int→float é automática; float→int só com
//!     cast explícito `int(x)`
//!   - condições de if/for exigem bool (sem truthiness)
//!   - strings são comparáveis (==/!=) — internadas no codegen
//!   - enums e structs são tipos opacos: enum só se compara com enum,
//!     struct não é comparável
//!   - `break`/`continue` só dentro de loop; `return` bate com o tipo
//!     declarado; métodos exigem receiver endereçável (variável)

use crate::ast::*;
use crate::lexer::Span;
use std::collections::HashMap;

#[derive(Debug)]
pub struct CheckError {
    pub msg: String,
    pub span: Span,
}

impl std::fmt::Display for CheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: erro de tipo: {}", self.span.line, self.span.col, self.msg)
    }
}

fn err(span: Span, msg: impl Into<String>) -> CheckError {
    CheckError { msg: msg.into(), span }
}

/// Assinatura resolvida de uma função.
#[derive(Clone)]
struct FuncSig {
    params: Vec<Type>,
    ret: Option<Type>,
}

pub struct Checker {
    /// struct → campo → tipo.
    structs: HashMap<String, HashMap<String, Type>>,
    /// enum → variantes (ordem = valor).
    enums: HashMap<String, Vec<String>>,
    /// símbolo da função (métodos: "Tipo.nome") → assinatura.
    funcs: HashMap<String, FuncSig>,
    /// Escopos de variáveis: o último é o mais interno.
    scopes: Vec<HashMap<String, Type>>,
    /// Profundidade de loops (break/continue).
    loop_depth: u32,
}

impl Checker {
    pub fn new() -> Self {
        Self {
            structs: HashMap::new(),
            enums: HashMap::new(),
            funcs: HashMap::new(),
            scopes: vec![HashMap::new()],
            loop_depth: 0,
        }
    }

    // ======================= programa inteiro =========================

    /// Ponto de entrada: programa → programa verificado (com casts).
    pub fn check_program(&mut self, program: &Program) -> Result<Program, CheckError> {
        self.collect(program)?;
        self.check_cycles()?;

        let mut decls = Vec::new();
        for decl in &program.decls {
            match decl {
                Decl::Func(f) => decls.push(Decl::Func(self.check_func(f)?)),
                other => decls.push(other.clone()),
            }
        }
        Ok(Program { imports: program.imports.clone(), decls })
    }

    /// Coleta structs/enums/funções e valida as assinaturas.
    /// Dois passos para structs: primeiro registra todas (referências
    /// para frente entre structs são válidas), depois valida campos.
    fn collect(&mut self, program: &Program) -> Result<(), CheckError> {
        // Passo A: registra enums e structs.
        for decl in &program.decls {
            match decl {
                Decl::Enum(e) => {
                    self.enums.insert(e.name.clone(), e.variants.clone());
                }
                Decl::Struct(s) => {
                    let fields: HashMap<String, Type> =
                        s.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect();
                    self.structs.insert(s.name.clone(), fields);
                }
                Decl::Func(_) => {}
            }
        }
        // Passo B: valida os tipos dos campos (todos os structs/enums já
        // são conhecidos agora).
        for decl in &program.decls {
            if let Decl::Struct(s) = decl {
                for f in &s.fields {
                    self.check_type_usable(&f.ty, f.span)?;
                }
            }
        }

        for decl in &program.decls {
            if let Decl::Func(f) = decl {
                let symbol = func_symbol(f);
                let mut params = Vec::new();
                if let Some(r) = &f.receiver {
                    if !self.structs.contains_key(&type_name(&r.ty)) {
                        return Err(err(
                            f.span,
                            format!(
                                "receiver deve ser de tipo struct, encontrou '{}'",
                                type_name(&r.ty)
                            ),
                        ));
                    }
                    params.push(r.ty.clone());
                }
                for p in &f.params {
                    self.check_type_usable(&p.ty, f.span)?;
                    params.push(p.ty.clone());
                }
                // ret: Some(void) é aceito e normalizado para None.
                let ret = match &f.ret {
                    None => None,
                    Some(Type::Void) => None,
                    Some(t) => {
                        self.check_type_usable(t, f.span)?;
                        Some(t.clone())
                    }
                };
                self.funcs.insert(symbol, FuncSig { params, ret });
            }
        }
        Ok(())
    }

    /// Tipo usado em posição de valor (campo, parâmetro, retorno):
    /// primitivo ou tipo nomeado declarado; void só como retorno explícito.
    fn check_type_usable(&self, ty: &Type, span: Span) -> Result<(), CheckError> {
        match ty {
            Type::Int | Type::Float | Type::Bool | Type::Str => Ok(()),
            Type::Void => Err(err(span, "tipo 'void' só é válido como retorno de função")),
            Type::Named(n) if self.structs.contains_key(n) || self.enums.contains_key(n) => Ok(()),
            Type::Named(n) => Err(err(span, format!("tipo '{n}' desconhecido"))),
        }
    }

    /// Ciclos de struct por valor (a→b→a) são inválidos (F5: ponteiros).
    fn check_cycles(&self) -> Result<(), CheckError> {
        for name in self.structs.keys() {
            let mut path: Vec<String> = Vec::new();
            if let Some(cycle) = self.struct_cycle(name, &mut path) {
                return Err(err(
                    Span::new(0, 0),
                    format!("struct recursiva por valor não suportada (F5: ponteiros): {cycle}"),
                ));
            }
        }
        Ok(())
    }

    fn struct_cycle(&self, start: &str, path: &mut Vec<String>) -> Option<String> {
        if let Some(pos) = path.iter().position(|p| p == start) {
            let mut cycle: Vec<&str> = path[pos..].iter().map(|s| s.as_str()).collect();
            cycle.push(start);
            return Some(cycle.join(" → "));
        }
        let fields = self.structs.get(start)?;
        path.push(start.to_string());
        for ty in fields.values() {
            if let Type::Named(n) = ty {
                if self.structs.contains_key(n) {
                    if let Some(cycle) = self.struct_cycle(n, path) {
                        return Some(cycle);
                    }
                }
            }
        }
        path.pop();
        None
    }

    // ============================ funções =============================

    fn check_func(&mut self, f: &FuncDecl) -> Result<FuncDecl, CheckError> {
        let ret = match &f.ret {
            Some(Type::Void) => None,
            other => other.clone(),
        };
        self.scopes.clear();
        self.scopes.push(HashMap::new());
        self.loop_depth = 0;

        if let Some(r) = &f.receiver {
            self.declare(&r.name, &r.ty, r.ty == Type::Void);
        }
        for p in &f.params {
            self.declare(&p.name, &p.ty, false);
        }

        let body = self.check_stmts(&f.body, &ret)?;
        Ok(FuncDecl {
            receiver: f.receiver.clone(),
            name: f.name.clone(),
            params: f.params.clone(),
            ret,
            body,
            span: f.span,
        })
    }

    fn declare(&mut self, name: &str, ty: &Type, is_void: bool) {
        let _ = is_void;
        self.scopes
            .last_mut()
            .expect("escopo raiz")
            .insert(name.to_string(), ty.clone());
    }

    // =========================== statements ===========================

    fn check_stmts(&mut self, stmts: &[Stmt], ret: &Option<Type>) -> Result<Vec<Stmt>, CheckError> {
        let mut out = Vec::new();
        for s in stmts {
            out.push(self.check_stmt(s, ret)?);
        }
        Ok(out)
    }

    /// Statements de bloco: empurra um escopo novo para `let` locais.
    fn check_block(&mut self, stmts: &[Stmt], ret: &Option<Type>) -> Result<Vec<Stmt>, CheckError> {
        self.scopes.push(HashMap::new());
        let out = self.check_stmts(stmts, ret);
        self.scopes.pop();
        out
    }

    fn check_stmt(&mut self, stmt: &Stmt, ret: &Option<Type>) -> Result<Stmt, CheckError> {
        match stmt {
            Stmt::Let { name, ty, value, span } => {
                let (value, vt) = self.check_expr(value)?;
                let expected = match ty {
                    Some(t) => {
                        self.check_type_usable(t, *span)?;
                        t.clone()
                    }
                    None => vt.clone(),
                };
                let value = self.coerce(value, &vt, &expected, *span)?;
                self.scopes
                    .last_mut()
                    .expect("escopo raiz")
                    .insert(name.clone(), expected);
                Ok(Stmt::Let { name: name.clone(), ty: ty.clone(), value, span: *span })
            }
            Stmt::Return(value, span) => match (value, ret) {
                (Some(v), Some(expected)) => {
                    let (v, vt) = self.check_expr(v)?;
                    let v = self.coerce(v, &vt, expected, *span)?;
                    Ok(Stmt::Return(Some(v), *span))
                }
                (Some(_), None) => Err(err(*span, "função void não pode retornar valor")),
                (None, Some(t)) => Err(err(
                    *span,
                    format!("função retorna '{}', mas 'return;' não tem valor", type_name(t)),
                )),
                (None, None) => Ok(Stmt::Return(None, *span)),
            },
            Stmt::If { cond, then_body, else_body, span } => {
                let (cond, ct) = self.check_expr(cond)?;
                if ct != Type::Bool {
                    return Err(err(
                        cond.span(),
                        format!("condição deve ser bool, encontrou '{}'", type_name(&ct)),
                    ));
                }
                let then_body = self.check_block(then_body, ret)?;
                let else_body = match else_body {
                    Some(b) => Some(self.check_block(b, ret)?),
                    None => None,
                };
                Ok(Stmt::If { cond, then_body, else_body, span: *span })
            }
            Stmt::For { init, cond, post, body, span } => {
                self.scopes.push(HashMap::new());
                let init = match init {
                    Some(i) => Some(Box::new(self.check_stmt(i.as_ref(), ret)?)),
                    None => None,
                };
                let cond = match cond {
                    Some(c) => {
                        let (c, ct) = self.check_expr(c)?;
                        if ct != Type::Bool {
                            return Err(err(
                                c.span(),
                                format!("condição deve ser bool, encontrou '{}'", type_name(&ct)),
                            ));
                        }
                        Some(c)
                    }
                    None => None,
                };
                let post = match post {
                    Some(p) => Some(Box::new(self.check_stmt(p.as_ref(), ret)?)),
                    None => None,
                };
                self.loop_depth += 1;
                let body = self.check_stmts(body, ret);
                self.loop_depth -= 1;
                let body = body?;
                self.scopes.pop();
                Ok(Stmt::For { init, cond, post, body, span: *span })
            }
            Stmt::Break(span) => {
                if self.loop_depth == 0 {
                    return Err(err(*span, "break fora de loop"));
                }
                Ok(Stmt::Break(*span))
            }
            Stmt::Continue(span) => {
                if self.loop_depth == 0 {
                    return Err(err(*span, "continue fora de loop"));
                }
                Ok(Stmt::Continue(*span))
            }
            Stmt::Assign { target, value, span } => {
                let (target, tt) = self.check_lvalue(target)?;
                let (value, vt) = self.check_expr(value)?;
                let value = self.coerce(value, &vt, &tt, *span)?;
                Ok(Stmt::Assign { target, value, span: *span })
            }
            Stmt::Expr(e) => {
                // Chamada como statement: void é permitido (valor descartado).
                if matches!(e, Expr::Call { .. }) {
                    let (e, _) = self.check_call_allow_void(e)?;
                    Ok(Stmt::Expr(e))
                } else {
                    let (e, _) = self.check_expr(e)?;
                    Ok(Stmt::Expr(e))
                }
            }
        }
    }

    /// Lvalue: variável ou cadeia de campos começando em variável.
    fn check_lvalue(&mut self, expr: &Expr) -> Result<(Expr, Type), CheckError> {
        match expr {
            Expr::Ident(name, span) => {
                let ty = self.lookup(name, *span)?;
                Ok((expr.clone(), ty))
            }
            Expr::Member { obj, field, span } => {
                let (obj, oty) = self.check_lvalue(obj)?;
                let ty = self.field_type(&obj, &oty, field, *span)?;
                Ok((
                    Expr::Member { obj: Box::new(obj), field: field.clone(), span: *span },
                    ty,
                ))
            }
            other => Err(err(
                other.span(),
                "alvo de atribuição deve ser variável ou campo de variável",
            )),
        }
    }

    fn lookup(&self, name: &str, span: Span) -> Result<Type, CheckError> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Ok(ty.clone());
            }
        }
        Err(err(span, format!("variável '{name}' não definida")))
    }

    // =========================== expressões ===========================

    fn check_expr(&mut self, expr: &Expr) -> Result<(Expr, Type), CheckError> {
        match expr {
            Expr::Int(n, s) => Ok((Expr::Int(*n, *s), Type::Int)),
            Expr::Float(n, s) => Ok((Expr::Float(*n, *s), Type::Float)),
            Expr::Bool(b, s) => Ok((Expr::Bool(*b, *s), Type::Bool)),
            Expr::Str(s, span) => Ok((Expr::Str(s.clone(), *span), Type::Str)),
            Expr::Ident(name, span) => {
                let ty = self.lookup(name, *span)?;
                Ok((expr.clone(), ty))
            }
            Expr::Unary { op, operand, span } => {
                let (operand, ot) = self.check_expr(operand)?;
                match op {
                    UnOp::Neg => match ot {
                        Type::Int | Type::Float => {
                            Ok((Expr::Unary { op: *op, operand: Box::new(operand), span: *span }, ot))
                        }
                        _ => Err(err(
                            *span,
                            format!("'-' exige int ou float, encontrou '{}'", type_name(&ot)),
                        )),
                    },
                    UnOp::Not => {
                        if ot != Type::Bool {
                            return Err(err(
                                *span,
                                format!("'!' exige bool, encontrou '{}'", type_name(&ot)),
                            ));
                        }
                        Ok((Expr::Unary { op: *op, operand: Box::new(operand), span: *span }, Type::Bool))
                    }
                }
            }
            Expr::Binary { op, lhs, rhs, span } => self.check_binary(*op, lhs, rhs, *span),
            Expr::Member { obj, field, span } => {
                // Enum value: `Mood.Stressed` — obj é o NOME do enum.
                if let Expr::Ident(name, _) = obj.as_ref() {
                    if let Some(variants) = self.enums.get(name) {
                        if !variants.iter().any(|v| v == field) {
                            return Err(err(
                                *span,
                                format!("enum '{name}' não tem variante '{field}'"),
                            ));
                        }
                        return Ok((
                            Expr::Member { obj: obj.clone(), field: field.clone(), span: *span },
                            Type::Named(name.clone()),
                        ));
                    }
                    if self.structs.contains_key(name) {
                        return Err(err(*span, format!("struct '{name}' não é um valor (use um literal ou variável)")));
                    }
                }
                let (obj, oty) = self.check_expr(obj)?;
                let ty = self.field_type(&obj, &oty, field, *span)?;
                Ok((Expr::Member { obj: Box::new(obj), field: field.clone(), span: *span }, ty))
            }
            Expr::StructLit { name, fields, span } => {
                // Clona a definição: check_expr precisa de &mut self.
                let def = self
                    .structs
                    .get(name)
                    .cloned()
                    .ok_or_else(|| err(*span, format!("struct '{name}' não declarada")))?;
                let mut out = Vec::new();
                let mut missing: Vec<String> = def.keys().cloned().collect();
                for (fname, fexpr) in fields {
                    let fty = def
                        .get(fname)
                        .ok_or_else(|| err(*span, format!("struct '{name}' não tem campo '{fname}'")))?
                        .clone();
                    let (fe, ft) = self.check_expr(fexpr)?;
                    let fe = self.coerce(fe, &ft, &fty, fexpr.span())?;
                    out.push((fname.clone(), fe));
                    missing.retain(|m| m != fname);
                }
                if let Some(fname) = missing.first() {
                    return Err(err(*span, format!("struct literal '{name}' sem o campo '{fname}'")));
                }
                Ok((Expr::StructLit { name: name.clone(), fields: out, span: *span }, Type::Named(name.clone())))
            }
            Expr::Call { callee, args, span } => self.check_call(callee, args, *span),
            Expr::Cast { to, expr, span } => {
                // Cast vindo de outra passada (idempotente): revalida.
                let (e, et) = self.check_expr(expr)?;
                let castable = matches!((to, &et), (Type::Int, Type::Int) | (Type::Int, Type::Float) | (Type::Float, Type::Float) | (Type::Float, Type::Int));
                if !castable {
                    return Err(err(*span, format!("não dá para converter '{}' para '{}'", type_name(&et), type_name(to))));
                }
                Ok((Expr::Cast { to: to.clone(), expr: Box::new(e), span: *span }, to.clone()))
            }
        }
    }

    /// Tipo do campo de um struct, validando que o objeto é struct.
    fn field_type(&self, _obj: &Expr, oty: &Type, field: &str, span: Span) -> Result<Type, CheckError> {
        let Type::Named(ty_name) = oty else {
            return Err(err(span, format!("tipo '{}' não é struct", type_name(oty))));
        };
        let def = self
            .structs
            .get(ty_name)
            .ok_or_else(|| err(span, format!("tipo '{}' não é struct", type_name(oty))))?;
        def.get(field)
            .cloned()
            .ok_or_else(|| err(span, format!("struct '{ty_name}' não tem campo '{field}'")))
    }

    fn check_binary(&mut self, op: BinOp, lhs: &Expr, rhs: &Expr, span: Span) -> Result<(Expr, Type), CheckError> {
        use BinOp::*;
        match op {
            And | Or => {
                let (l, lt) = self.check_expr(lhs)?;
                let (r, rt) = self.check_expr(rhs)?;
                if lt != Type::Bool || rt != Type::Bool {
                    let bad = if lt != Type::Bool { &lt } else { &rt };
                    return Err(err(span, format!("'{}' exige bool, encontrou '{}'", op_symbol(op), type_name(bad))));
                }
                Ok((Expr::Binary { op, lhs: Box::new(l), rhs: Box::new(r), span }, Type::Bool))
            }
            Eq | Ne => {
                let (l, lt) = self.check_expr(lhs)?;
                let (r, rt) = self.check_expr(rhs)?;
                let (l, r, u) = self.unify(l, lt, r, rt, span)?;
                // Enums comparam entre si (valores); structs nunca.
                if let Type::Named(ty_name) = &u {
                    if self.structs.contains_key(ty_name) {
                        return Err(err(span, "structs não são comparáveis (F5)"));
                    }
                }
                Ok((Expr::Binary { op, lhs: Box::new(l), rhs: Box::new(r), span }, Type::Bool))
            }
            Lt | Le | Gt | Ge => {
                let (l, lt) = self.check_expr(lhs)?;
                let (r, rt) = self.check_expr(rhs)?;
                let (l, r, num) = self.unify(l, lt, r, rt, span)?;
                if !matches!(num, Type::Int | Type::Float) {
                    return Err(err(span, format!("'{}' exige int ou float, encontrou '{}'", op_symbol(op), type_name(&num))));
                }
                Ok((Expr::Binary { op, lhs: Box::new(l), rhs: Box::new(r), span }, Type::Bool))
            }
            Add | Sub | Mul | Div => {
                let (l, lt) = self.check_expr(lhs)?;
                let (r, rt) = self.check_expr(rhs)?;
                let (l, r, num) = self.unify(l, lt, r, rt, span)?;
                if !matches!(num, Type::Int | Type::Float) {
                    return Err(err(span, format!("'{}' exige int ou float, encontrou '{}'", op_symbol(op), type_name(&num))));
                }
                Ok((Expr::Binary { op, lhs: Box::new(l), rhs: Box::new(r), span }, num))
            }
        }
    }

    /// Iguala os tipos de dois operandos: tipos iguais passam, int+float
    /// promove o int para float (cast inserido). Retorna os operandos
    /// corrigidos e o tipo unificado.
    fn unify(
        &mut self,
        l: Expr,
        lt: Type,
        r: Expr,
        rt: Type,
        span: Span,
    ) -> Result<(Expr, Expr, Type), CheckError> {
        if lt == rt {
            return Ok((l, r, lt));
        }
        if lt == Type::Int && rt == Type::Float {
            let l = self.coerce(l, &lt, &Type::Float, span)?;
            return Ok((l, r, Type::Float));
        }
        if lt == Type::Float && rt == Type::Int {
            let r = self.coerce(r, &rt, &Type::Float, span)?;
            return Ok((l, r, Type::Float));
        }
        Err(err(
            span,
            format!("tipos incompatíveis: '{}' e '{}'", type_name(&lt), type_name(&rt)),
        ))
    }

    fn check_call(&mut self, callee: &Expr, args: &[Expr], span: Span) -> Result<(Expr, Type), CheckError> {
        self.check_call_inner(callee, args, span, false)
    }

    /// Chamada em posição de statement: void é aceito.
    fn check_call_allow_void(&mut self, expr: &Expr) -> Result<(Expr, Type), CheckError> {
        let Expr::Call { callee, args, span } = expr else {
            unreachable!()
        };
        self.check_call_inner(callee, args, *span, true)
    }

    fn check_call_inner(
        &mut self,
        callee: &Expr,
        args: &[Expr],
        span: Span,
        allow_void: bool,
    ) -> Result<(Expr, Type), CheckError> {
        // Casts: int(x) e float(x) são builtins (não são funções reais).
        if let Expr::Ident(name, _) = callee {
            if name == "int" || name == "float" {
                return self.check_cast_builtin(name, args, span);
            }
        }

        let (sig, arg_tys, _) = if let Expr::Member { obj, field, .. } = callee {
            // Método: receiver precisa ser endereçável (codegen passa por
            // referência) — variável ou campo de variável.
            let (_obj, oty) = self.check_lvalue(obj).map_err(|_| {
                err(
                    span,
                    "método exige receiver endereçável (variável) — atribua o valor a uma variável primeiro",
                )
            })?;
            let Type::Named(ty_name) = &oty else {
                return Err(err(span, format!("método exige receiver struct, encontrou '{}'", type_name(&oty))));
            };
            let symbol = format!("{ty_name}.{field}");
            let sig = self
                .funcs
                .get(&symbol)
                .ok_or_else(|| err(span, format!("tipo '{ty_name}' não tem método '{field}'")))?
                .clone();
            let arg_tys = vec![oty];
            (sig, arg_tys, ())
        } else {
            let Expr::Ident(fname, _) = callee else {
                return Err(err(span, "callee de chamada inválido"));
            };
            let sig = self
                .funcs
                .get(fname)
                .ok_or_else(|| err(span, format!("função '{fname}' não encontrada")))?
                .clone();
            (sig, Vec::new(), ())
        };

        // Aridade. Métodos: o receiver é implícito (derivado do callee).
        if args.len() + arg_tys.len() != sig.params.len() {
            return Err(err(
                span,
                format!(
                    "função esperava {} argumentos, recebeu {}",
                    sig.params.len() - arg_tys.len(),
                    args.len()
                ),
            ));
        }

        // Args com coerção para os tipos dos parâmetros. O receiver
        // NÃO entra aqui: o codegen o deriva do callee (Member.obj).
        let mut checked_args = Vec::new();
        for (i, a) in args.iter().enumerate() {
            let (a, at) = self.check_expr(a)?;
            let span = a.span();
            let pty = &sig.params[arg_tys.len() + i];
            let a = self.coerce(a, &at, pty, span)?;
            checked_args.push(a);
        }

        match &sig.ret {
            Some(t) => {
                let ret = t.clone();
                let callee = callee.clone();
                Ok((Expr::Call { callee: Box::new(callee), args: checked_args, span }, ret))
            }
            None if allow_void => {
                let callee = callee.clone();
                Ok((Expr::Call { callee: Box::new(callee), args: checked_args, span }, Type::Void))
            }
            None => Err(err(span, "função retorna void, mas o valor está sendo usado")),
        }
    }

    /// `int(x)` (float→int trunca) e `float(x)` (int→float).
    fn check_cast_builtin(&mut self, name: &str, args: &[Expr], span: Span) -> Result<(Expr, Type), CheckError> {
        let [a] = args else {
            return Err(err(span, format!("{name}(x) esperava exatamente 1 argumento")));
        };
        let (a, at) = self.check_expr(a)?;
        let to = if name == "int" { Type::Int } else { Type::Float };
        if !matches!(at, Type::Int | Type::Float) {
            return Err(err(span, format!("{name}(x) exige int ou float, encontrou '{}'", type_name(&at))));
        }
        let cast = if at == to {
            a // conversão identidade: sem node
        } else {
            Expr::Cast { to: to.clone(), expr: Box::new(a), span }
        };
        Ok((cast, to))
    }

    /// Coerção assimétrica: int→float automática; o resto só com cast.
    fn coerce(&mut self, expr: Expr, from: &Type, to: &Type, span: Span) -> Result<Expr, CheckError> {
        if from == to {
            return Ok(expr);
        }
        if *from == Type::Int && *to == Type::Float {
            return Ok(Expr::Cast { to: Type::Float, expr: Box::new(expr), span });
        }
        Err(err(
            span,
            format!(
                "esperava '{}', encontrou '{}'{}",
                type_name(to),
                type_name(from),
                if *from == Type::Float && *to == Type::Int {
                    " (float→int exige cast explícito: int(x))"
                } else {
                    ""
                }
            ),
        ))
    }
}

// ============================ helpers =================================

/// Símbolo da função no módulo (métodos ganham prefixo do tipo).
fn func_symbol(f: &FuncDecl) -> String {
    match &f.receiver {
        Some(r) => format!("{}.{}", type_name(&r.ty), f.name),
        None => f.name.clone(),
    }
}

fn op_symbol(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::Lt => "<",
        BinOp::Le => "<=",
        BinOp::Gt => ">",
        BinOp::Ge => ">=",
        BinOp::And => "&&",
        BinOp::Or => "||",
    }
}

// ========================== Conveniência =============================

/// Texto de programa → AST verificado (parse + check).
pub fn check_program(program: &Program) -> Result<Program, CheckError> {
    Checker::new().check_program(program)
}

/// Expressão isolada (caminho F1) → expressão verificada.
pub fn check_expr(expr: &Expr) -> Result<Expr, CheckError> {
    let mut c = Checker::new();
    c.check_expr(expr).map(|(e, _)| e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_program;

    fn err_of(src: &str) -> String {
        let program = parse_program(src).unwrap();
        check_program(&program).unwrap_err().msg
    }

    fn ok(src: &str) {
        let program = parse_program(src).unwrap();
        check_program(&program).unwrap();
    }

    // ---- tipos primitivos e coerção ----

    #[test]
    fn int_and_float_literals() {
        ok("func main(): float { let a = 42; let b = 42.0; return b; }");
    }

    #[test]
    fn implicit_int_to_float_is_allowed() {
        ok(r#"
            func add(a: float, b: float): float { return a + b; }
            func main(): float {
                let x: float = 1;          // let anotado
                let y = 2.5 + 1;           // binário misto
                return add(1, 2) + x + y;  // argumento de chamada
            }
        "#);
    }

    #[test]
    fn float_to_int_requires_explicit_cast() {
        let e = err_of("func main(): int { let x = 1.5; return x; }");
        assert!(e.contains("cast"), "{e}");
        let e = err_of("func main(): int { return 1.5; }");
        assert!(e.contains("cast"), "{e}");
        ok("func main(): int { return int(1.5); }");
    }

    #[test]
    fn int_division_is_valid() {
        ok("func main(): int { return 7 / 2; }");
    }

    #[test]
    fn type_mismatch_in_let_annotation() {
        let e = err_of("func main(): float { let x: int = 1.5; return 1.0; }");
        assert!(e.contains("int(x)"), "{e}");
    }

    #[test]
    fn arithmetic_on_bool_is_error() {
        assert!(err_of("func main(): float { return true + 1; }").contains("bool"));
        assert!(err_of("func main(): float { return 1 - false; }").contains("bool"));
    }

    #[test]
    fn comparison_type_mismatch_is_error() {
        let e = err_of("func main(): float { if 1.0 == true { return 1.0; } return 0.0; }");
        assert!(e.contains("incompatíveis"), "{e}");
    }

    #[test]
    fn condition_must_be_bool() {
        let e = err_of("func main(): float { if 1.0 { return 1.0; } return 0.0; }");
        assert!(e.contains("bool"), "{e}");
    }

    #[test]
    fn logic_requires_bool() {
        assert!(err_of("func main(): float { let x = 1.0 && true; return 1.0; }").contains("bool"));
    }

    // ---- strings ----

    #[test]
    fn strings_are_comparable() {
        ok(r#"
            func main(): float {
                let a = "oi";
                let b = "oi";
                if a == b && a != "tchau" { return 1.0; }
                return 0.0;
            }
        "#);
    }

    #[test]
    fn string_concat_is_error() {
        assert!(err_of(r#"func main(): float { let s = "a" + "b"; return 1.0; }"#).contains("string"));
        assert!(err_of(r#"func main(): float { if "a" < "b" { return 1.0; } return 0.0; }"#).contains("string"));
    }

    // ---- enums e structs ----

    #[test]
    fn enums_only_compare_to_same_enum() {
        ok("enum M { A, B } func main(): float { let m = M.A; if m == M.B { return 1.0; } return 0.0; }");
        assert!(err_of("enum M { A } func main(): float { if M.A == 1.0 { return 1.0; } return 0.0; }").contains("incompatíveis"));
        assert!(err_of("enum M { A } func main(): float { if M.A == 1 { return 1.0; } return 0.0; }").contains("incompatíveis"));
    }

    #[test]
    fn structs_are_not_comparable() {
        let e = err_of(r#"
            struct P { x: float; }
            func main(): float { let a = P { x: 1.0 }; let b = P { x: 2.0 }; if a == b { return 1.0; } return 0.0; }
        "#);
        assert!(e.contains("comparáveis"), "{e}");
    }

    #[test]
    fn struct_type_used_as_value_is_error() {
        let e = err_of("struct P { x: float; } func main(): float { return P.x; }");
        assert!(e.contains("não é um valor"), "{e}");
    }

    #[test]
    fn unknown_type_in_field_and_param() {
        let e = err_of("struct A { x: banana; } func main(): float { return 1.0; }");
        assert!(e.contains("banana"), "{e}");
        let e = err_of("func f(a: banana) { } func main(): float { return 1.0; }");
        assert!(e.contains("banana"), "{e}");
    }

    #[test]
    fn void_param_is_error() {
        assert!(err_of("func f(a: void) { } func main(): float { return 1.0; }").contains("void"));
    }

    // ---- retorno, chamadas, escopo ----

    #[test]
    fn void_return_with_value_is_error() {
        let e = err_of("func main() { return 1.0; }");
        assert!(e.contains("void"), "{e}");
    }

    #[test]
    fn non_void_return_without_value_is_error() {
        let e = err_of("func main(): float { return; }");
        assert!(e.contains("float"), "{e}");
    }

    #[test]
    fn call_arg_count_is_checked() {
        let e = err_of("func f(a: float) { } func main() { f(1.0, 2.0); }");
        assert!(e.contains("1 argumento"), "{e}");
    }

    #[test]
    fn unknown_function_is_error() {
        assert!(err_of("func main() { nope(1.0); }").contains("nope"));
    }

    #[test]
    fn using_void_result_is_error() {
        let e = err_of("func f() { } func main(): float { return f(); }");
        assert!(e.contains("void"), "{e}");
    }

    #[test]
    fn method_receiver_must_be_addressable() {
        let e = err_of(r#"
            struct P { x: float; }
            func (p: P) get(): float { return p.x; }
            func main(): float { return P { x: 1.0 }.get(); }
        "#);
        assert!(e.contains("endereçável"), "{e}");
    }

    #[test]
    fn method_on_non_struct_is_error() {
        let e = err_of("func main(): float { let x = 1.0; return x.field; }");
        assert!(e.contains("struct"), "{e}");
    }

    #[test]
    fn receiver_type_must_be_struct() {
        let e = err_of("func (x: float) f() { } func main(): float { return 1.0; }");
        assert!(e.contains("receiver"), "{e}");
    }

    #[test]
    fn unknown_member_and_variant() {
        let e = err_of("struct P { x: float; } func main(): float { let p = P { x: 1.0 }; return p.nope; }");
        assert!(e.contains("nope"), "{e}");
        let e = err_of("enum M { A } func main(): float { return M.B; }");
        assert!(e.contains("B"), "{e}");
    }

    #[test]
    fn struct_literal_field_checks() {
        assert!(err_of("struct P { x: float; y: float; } func main(): float { let p = P { x: 1.0 }; return p.x; }").contains("y"));
        assert!(err_of("struct P { x: float; } func main(): float { let p = P { x: 1.0, z: 2.0 }; return p.x; }").contains("z"));
    }

    #[test]
    fn break_continue_require_loop() {
        assert!(err_of("func main() { break; }").contains("loop"));
        assert!(err_of("func main() { continue; }").contains("loop"));
    }

    #[test]
    fn block_scope_ends() {
        let e = err_of("func main(): float { if true { let x = 1.0; } return x; }");
        assert!(e.contains("não definida"), "{e}");
    }

    #[test]
    fn for_loop_variable_scoped_to_loop() {
        let e = err_of("func main(): float { for let i = 0; i < 3; i = i + 1 { } return i; }");
        assert!(e.contains("não definida"), "{e}");
    }

    #[test]
    fn shadowing_is_allowed() {
        ok("func main(): float { let x = 1.0; let x = 2.0; return x; }");
    }

    #[test]
    fn recursive_struct_is_error() {
        let e = err_of("struct A { b: B; } struct B { a: A; } func main(): float { return 1.0; }");
        assert!(e.contains("recursiva"), "{e}");
    }

    #[test]
    fn casts_inserted_for_implicit_coercion() {
        let program = parse_program("func main(): float { return 1 + 2.5; }").unwrap();
        let checked = check_program(&program).unwrap();
        let Decl::Func(f) = &checked.decls[0] else { panic!() };
        let Stmt::Return(Some(e), _) = &f.body[0] else { panic!() };
        assert!(matches!(
            e,
            Expr::Binary { lhs, rhs, .. }
                if matches!(lhs.as_ref(), Expr::Cast { to: Type::Float, .. })
                    && matches!(rhs.as_ref(), Expr::Float(2.5, _))
        ));
    }
}
