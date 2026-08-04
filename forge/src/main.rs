//! forge — linguagem de script para game engine, compilada via LLVM.
//!
//! Pipeline: texto → lexer → parser → AST → codegen → JIT → execução
//!
//! Uso:
//!   forge "1 + 2 * 3"          → avalia expressão via JIT (F1)
//!   forge run <arquivo.forge>  → executa func main() via JIT (F2b)
//!   forge fmt <arquivo.forge>  → formata o arquivo (estilo canônico)
//!   forge ast <arquivo.forge>  → dump da AST (debug do parser)

mod ast;
mod checker;
mod codegen;
mod formatter;
mod lexer;
mod parser;

use codegen::MainResult;
use inkwell::context::Context;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.len() {
        2 => match run_expr(&args[1]) {
            Ok(result) => println!("{} = {result}", args[1]),
            Err(e) => die(&e),
        },
        3 => match args[1].as_str() {
            "run" => run_file(&args[2]),
            "fmt" => run_fmt(&args[2]),
            "ast" => run_ast(&args[2]),
            other => die(&format!("subcomando desconhecido: '{other}'")),
        },
        _ => help(),
    }
}

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}

fn help() -> ! {
    eprintln!("uso:");
    eprintln!("  forge \"<expressão>\"          avalia expressão via LLVM JIT");
    eprintln!("  forge run <arquivo.forge>     executa func main() via JIT");
    eprintln!("  forge fmt <arquivo.forge>     formata o arquivo");
    eprintln!("  forge ast <arquivo.forge>     mostra a AST parseada");
    std::process::exit(1);
}

/// Caminho F1: expressão pura → type check → JIT.
fn run_expr(src: &str) -> Result<f64, String> {
    let expr = parser::parse_expr(src).map_err(|e| e.to_string())?;
    let expr = checker::check_expr(&expr).map_err(|e| e.to_string())?;
    let context = Context::create();
    let mut cg = codegen::Codegen::new(&context);
    cg.compile_top(&expr)?;
    cg.run_jit()
}

/// `forge run <file>`: parseia, verifica tipos e executa `main` via JIT.
fn run_file(path: &str) {
    let src = std::fs::read_to_string(path)
        .unwrap_or_else(|e| die(&format!("não consegui ler '{path}': {e}")));
    let program = match parser::parse_program(&src) {
        Ok(p) => p,
        Err(e) => die(&e.to_string()),
    };
    let program = match checker::check_program(&program) {
        Ok(p) => p,
        Err(e) => die(&e.to_string()),
    };
    let context = Context::create();
    let mut cg = codegen::Codegen::new(&context);
    if let Err(e) = cg.compile_program(&program) {
        die(&e);
    }
    match cg.run_main() {
        Ok(MainResult::Float(v)) => println!("main() = {v}"),
        Ok(MainResult::Int(v)) => println!("main() = {v}"),
        Ok(MainResult::Void) => println!("main() executada (void)"),
        Err(e) => die(&e),
    }
}

/// `forge fmt <file>`: parse + formatação canônica.
fn run_fmt(path: &str) {
    let src = std::fs::read_to_string(path)
        .unwrap_or_else(|e| die(&format!("não consegui ler '{path}': {e}")));
    match parser::parse_program(&src) {
        Ok(program) => print!("{}", formatter::format_program(&program)),
        Err(e) => die(&e.to_string()),
    }
}

/// `forge ast <file>`: dump da AST para inspecionar o parse.
fn run_ast(path: &str) {
    let src = std::fs::read_to_string(path)
        .unwrap_or_else(|e| die(&format!("não consegui ler '{path}': {e}")));
    match parser::parse_program(&src) {
        Ok(program) => println!("{:#?}", program),
        Err(e) => die(&e.to_string()),
    }
}
