//! Codegen: AST → LLVM IR, e execução via JIT.
//!
//! F1:  expressão única → função `top()` → JIT
//! F2b: programa inteiro → declara todas as funções (forward refs),
//!      gera corpos (Alloca/Load/Store p/ variáveis), `main` como entry.
//!
//! Tipos: por enquanto tudo é f64 (como o Kaleidoscope). O type checker
//! (F4) traz int/bool/string. Structs + receiver + member access = F2c.

use crate::ast::*;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::module::Module;
use inkwell::types::BasicMetadataTypeEnum;
use inkwell::values::{BasicMetadataValueEnum, BasicValueEnum, FloatValue, PointerValue};
use inkwell::OptimizationLevel;
use std::collections::HashMap;

/// Assinatura da função top-level do caminho F1 (expressão).
pub type TopFn = unsafe extern "C" fn() -> f64;

pub struct Codegen<'ctx> {
    pub context: &'ctx Context,
    pub module: Module<'ctx>,
    pub builder: Builder<'ctx>,
    /// Símbolo → retorna float? (para saber a assinatura JIT de main)
    ret_is_float: HashMap<String, bool>,
    /// Variáveis locais da função corrente: nome → alloca.
    vars: HashMap<String, PointerValue<'ctx>>,
}

impl<'ctx> Codegen<'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
        let module = context.create_module("forge");
        let builder = context.create_builder();
        Self {
            context,
            module,
            builder,
            ret_is_float: HashMap::new(),
            vars: HashMap::new(),
        }
    }

    // ========================= F2b: programa ==========================

    /// Compila um programa inteiro. Duas passadas: primeiro declara todas
    /// as funções (permite forward references), depois gera os corpos.
    pub fn compile_program(&mut self, program: &Program) -> Result<(), String> {
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

    /// Símbolo da função no módulo. Métodos (receiver) ganham prefixo do
    /// tipo para evitar colisão: `Citizen.go` — a F2c resolve member calls
    /// por este símbolo.
    fn func_symbol(f: &FuncDecl) -> String {
        match &f.receiver {
            Some(r) => format!("{}.{}", type_name(&r.ty), f.name),
            None => f.name.clone(),
        }
    }

    fn check_type(&self, t: &Type) -> Result<(), String> {
        match t {
            Type::Named(n) if n == "float" => Ok(()),
            Type::Named(n) => Err(format!(
                "tipo '{n}' ainda não suportado no codegen (F4) — por enquanto só 'float'"
            )),
        }
    }

    /// Cria o cabeçalho da função no módulo (sem corpo).
    fn declare_func(&mut self, f: &FuncDecl) -> Result<(), String> {
        let f64 = self.context.f64_type();
        let void = self.context.void_type();

        // Params: receiver (se houver) + params declarados.
        let mut param_tys: Vec<BasicMetadataTypeEnum> = Vec::new();
        if let Some(r) = &f.receiver {
            self.check_type(&r.ty)?;
            param_tys.push(f64.into());
        }
        for p in &f.params {
            self.check_type(&p.ty)?;
            param_tys.push(f64.into());
        }

        let ret_float = match &f.ret {
            None => false,
            Some(t) => {
                self.check_type(t)?;
                true
            }
        };
        let fn_ty = if ret_float {
            f64.fn_type(&param_tys, false)
        } else {
            void.fn_type(&param_tys, false)
        };
        let symbol = Self::func_symbol(f);
        self.ret_is_float.insert(symbol.clone(), ret_float);
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

        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        // Params viram variáveis locais (alloca + store) — padrão Kaleidoscope.
        let mut idx = 0u32;
        if let Some(r) = &f.receiver {
            let p = function
                .get_nth_param(idx)
                .ok_or_else(|| "parametro receiver faltando".to_string())?;
            self.assign_param(&r.name, p)?;
            idx += 1;
        }
        for p in &f.params {
            let param = function
                .get_nth_param(idx)
                .ok_or_else(|| format!("parametro '{}' faltando", p.name))?;
            self.assign_param(&p.name, param)?;
            idx += 1;
        }

        // Corpo. Após um `return` explícito, o resto é inalcançável — para
        // de gerar (F3 trata returns dentro de blocos com mais cuidado).
        let mut ended = false;
        for stmt in &f.body {
            if ended {
                break;
            }
            ended = matches!(stmt, Stmt::Return(..));
            self.gen_stmt(stmt)?;
        }

        // Retorno implícito: 0.0 se a função retorna float, void caso contrário.
        if !ended {
            match &f.ret {
                Some(_) => {
                    let zero = self.context.f64_type().const_float(0.0);
                    self.builder.build_return(Some(&zero)).map_err(|e| e.to_string())?;
                }
                None => {
                    self.builder.build_return(None).map_err(|e| e.to_string())?;
                }
            }
        }
        Ok(())
    }

    fn assign_param(&mut self, name: &str, param: BasicValueEnum<'ctx>) -> Result<(), String> {
        let ptr = self
            .builder
            .build_alloca(self.context.f64_type(), name)
            .map_err(|e| e.to_string())?;
        self.builder.build_store(ptr, param).map_err(|e| e.to_string())?;
        self.vars.insert(name.to_string(), ptr);
        Ok(())
    }

    // ----------------------- statements -------------------------------

    fn gen_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::Let { name, ty, value, .. } => {
                if let Some(t) = ty {
                    self.check_type(t)?;
                }
                let val = self.gen_expr(value)?;
                let ptr = self
                    .builder
                    .build_alloca(self.context.f64_type(), name)
                    .map_err(|e| e.to_string())?;
                self.builder.build_store(ptr, val).map_err(|e| e.to_string())?;
                self.vars.insert(name.clone(), ptr);
                Ok(())
            }
            Stmt::Return(value, _) => {
                match value {
                    Some(v) => {
                        let val = self.gen_expr(v)?;
                        self.builder.build_return(Some(&val)).map_err(|e| e.to_string())?;
                    }
                    None => {
                        self.builder.build_return(None).map_err(|e| e.to_string())?;
                    }
                }
                Ok(())
            }
            Stmt::Expr(e) => {
                // Side effects (chamadas) são o que importa; valor é descartado.
                self.gen_expr(e)?;
                Ok(())
            }
        }
    }

    // ----------------------- expressões -------------------------------

    fn gen_expr(&self, expr: &Expr) -> Result<FloatValue<'ctx>, String> {
        match expr {
            Expr::Number(n, _) => Ok(self.context.f64_type().const_float(*n)),
            Expr::Ident(name, _) => {
                let ptr = self
                    .vars
                    .get(name)
                    .ok_or_else(|| format!("variável '{name}' não definida"))?;
                let v = self
                    .builder
                    .build_load(self.context.f64_type(), *ptr, name)
                    .map_err(|e| e.to_string())?;
                // F2b: só floats existem — o tipo é garantido pelo design.
                Ok(v.into_float_value())
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                let l = self.gen_expr(lhs)?;
                let r = self.gen_expr(rhs)?;
                match op {
                    BinOp::Add => self.builder.build_float_add(l, r, "addtmp"),
                    BinOp::Sub => self.builder.build_float_sub(l, r, "subtmp"),
                    BinOp::Mul => self.builder.build_float_mul(l, r, "multmp"),
                    BinOp::Div => self.builder.build_float_div(l, r, "divtmp"),
                }
                .map_err(|e| e.to_string())
            }
            Expr::Call { callee, args, .. } => {
                let Expr::Ident(fname, _) = callee.as_ref() else {
                    return Err("chamada de método (a.b()) ainda não suportada (F2c)".into());
                };
                let function = self
                    .module
                    .get_function(fname)
                    .ok_or_else(|| format!("função '{fname}' não encontrada"))?;
                let mut arg_vals: Vec<BasicMetadataValueEnum> = Vec::new();
                for a in args {
                    arg_vals.push(self.gen_expr(a)?.into());
                }
                let call = self
                    .builder
                    .build_call(function, &arg_vals, "calltmp")
                    .map_err(|e| e.to_string())?;
                match call.try_as_basic_value().basic() {
                    Some(v) => {
                        // F2b: só floats existem — o tipo é garantido pelo design.
                        Ok(v.into_float_value())
                    }
                    None => Err(format!(
                        "função '{fname}' retorna void, mas o valor está sendo usado"
                    )),
                }
            }
            Expr::Str(..) | Expr::Member { .. } => {
                Err("expressão ainda não suportada no codegen (strings: F4, membros: F2c)".into())
            }
        }
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

        match self.ret_is_float.get("main").copied() {
            Some(true) => {
                type MainF = unsafe extern "C" fn() -> f64;
                // SAFETY: main foi gerada por nós com assinatura compatível.
                let f: JitFunction<MainF> =
                    unsafe { engine.get_function("main") }.map_err(|e| e.to_string())?;
                unsafe { Ok(Some(f.call())) }
            }
            Some(false) => {
                type MainV = unsafe extern "C" fn();
                let f: JitFunction<MainV> =
                    unsafe { engine.get_function("main") }.map_err(|e| e.to_string())?;
                unsafe {
                    f.call();
                }
                Ok(None)
            }
            None => Err("'main' não foi declarada pelo codegen".into()),
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

fn type_name(ty: &Type) -> String {
    match ty {
        Type::Named(name) => name.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_expr;
    use crate::parser::parse_program;
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
        // main chama f, f chama g — g é declarada depois de f.
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
    fn params_are_local_variables() {
        let src = r#"
            func scale(x: float): float {
                let x = x * 2.0;   // shadowing do param — deve funcionar
                return x;
            }
            func main(): float {
                return scale(21.0);
            }
        "#;
        assert_eq!(eval_main(src), 42.0);
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
    fn implicit_return_zero() {
        // função float sem return explícito → 0.0 (Kaleidoscope default)
        let src = r#"
            func nothing(): float {
                let x = 42.0;
            }
            func main(): float {
                return nothing();
            }
        "#;
        assert_eq!(eval_main(src), 0.0);
    }

    #[test]
    fn expression_statement_works() {
        let src = r#"
            func f(x: float): float {
                return x + 1.0;
            }
            func main(): float {
                f(10.0);        // chamada como statement — valor descartado
                return 1.0;
            }
        "#;
        assert_eq!(eval_main(src), 1.0);
    }

    #[test]
    fn missing_main_is_error() {
        let src = "func f(): float { return 1.0; }";
        assert!(eval_program(src).is_err());
    }

    #[test]
    fn undefined_variable_is_error() {
        let src = "func main(): float { return x; }";
        let err = eval_program(src).unwrap_err();
        assert!(err.contains("x"));
    }

    #[test]
    fn unknown_type_is_error() {
        let src = "func main(): int { return 1; }";
        let err = eval_program(src).unwrap_err();
        assert!(err.contains("int"));
    }

    #[test]
    fn method_call_is_clear_error() {
        let src = "func main(): float { return a.b(); }";
        let err = eval_program(src).unwrap_err();
        assert!(err.contains("método"));
    }
}
