//! Formatter: AST → texto canônico (filosofia gofmt).
//!
//! O compilador é dono do estilo: uma única representação possível.
//! Regras:
//!   - indentação: 4 espaços
//!   - chaves em K&R (abrem na mesma linha)
//!   - `a + b` com espaços, `f(a, b)` sem
//!   - statements sempre terminam com `;`
//!
//! Garantias testadas: idempotência (fmt(fmt(x)) == fmt(x)) e
//! round-trip (parse(fmt(x)) == parse(x)).

use crate::ast::*;

pub struct Formatter {
    out: String,
    indent: u32,
}

const INDENT: &str = "    ";

impl Formatter {
    pub fn new() -> Self {
        Self {
            out: String::new(),
            indent: 0,
        }
    }

    fn line(&mut self) {
        self.out.push('\n');
        for _ in 0..self.indent {
            self.out.push_str(INDENT);
        }
    }

    fn push(&mut self, s: &str) {
        self.out.push_str(s);
    }

    fn with_indent<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.indent += 1;
        let r = f(self);
        self.indent -= 1;
        r
    }

    // ------------------------- programa ------------------------------

    pub fn format_program(program: &Program) -> String {
        let mut f = Formatter::new();
        f.program(program);
        // sempre termina com nova linha
        if !f.out.ends_with('\n') {
            f.out.push('\n');
        }
        f.out
    }

    fn program(&mut self, program: &Program) {
        let mut first = true;
        for import in &program.imports {
            if !first {
                self.line();
            }
            self.import(import);
            first = false;
        }
        for decl in &program.decls {
            // linha em branco entre declarações (e após imports)
            if !first {
                self.line();
                self.line();
            }
            self.decl(decl);
            first = false;
        }
    }

    fn import(&mut self, import: &Import) {
        self.push("import { ");
        for (i, name) in import.names.iter().enumerate() {
            if i > 0 {
                self.push(", ");
            }
            self.push(name);
        }
        self.push(&format!(" }} from \"{}\";", import.from));
    }

    fn decl(&mut self, decl: &Decl) {
        match decl {
            Decl::Enum(e) => self.enum_decl(e),
            Decl::Struct(s) => self.struct_decl(s),
            Decl::Func(f) => self.func_decl(f),
        }
    }

    fn enum_decl(&mut self, e: &EnumDecl) {
        self.push(&format!("enum {} {{", e.name));
        self.with_indent(|w| {
            for variant in &e.variants {
                w.line();
                w.push(&format!("{variant},"));
            }
        });
        self.line();
        self.push("}");
    }

    fn struct_decl(&mut self, s: &StructDecl) {
        self.push(&format!("struct {} {{", s.name));
        self.with_indent(|f| {
            for field in &s.fields {
                f.line();
                f.type_field(&field.name, &field.ty);
                f.push(";");
            }
        });
        self.line();
        self.push("}");
    }

    fn func_decl(&mut self, f: &FuncDecl) {
        self.push("func ");
        if let Some(receiver) = &f.receiver {
            self.push(&format!("({}: {}) ", receiver.name, type_name(&receiver.ty)));
        }
        self.push(&format!("{}(", f.name));
        for (i, p) in f.params.iter().enumerate() {
            if i > 0 {
                self.push(", ");
            }
            self.push(&format!("{}: {}", p.name, type_name(&p.ty)));
        }
        self.push(")");
        if let Some(ret) = &f.ret {
            self.push(&format!(": {}", type_name(ret)));
        }
        self.push(" {");
        self.with_indent(|w| {
            for stmt in &f.body {
                w.line();
                w.stmt(stmt);
            }
        });
        self.line();
        self.push("}");
    }

    fn type_field(&mut self, name: &str, ty: &Type) {
        self.push(&format!("{name}: {}", type_name(ty)));
    }

    // ------------------------- statements ----------------------------

    fn stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let { name, ty, value, .. } => {
                self.push("let ");
                self.push(name);
                if let Some(ty) = ty {
                    self.push(&format!(": {}", type_name(ty)));
                }
                self.push(" = ");
                self.expr(value);
                self.push(";");
            }
            Stmt::Return(value, _) => {
                self.push("return");
                if let Some(value) = value {
                    self.push(" ");
                    self.expr(value);
                }
                self.push(";");
            }
            Stmt::Expr(expr) => {
                self.expr(expr);
                self.push(";");
            }
        }
    }

    // ------------------------- expressões ----------------------------

    fn expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Number(n, _) => self.push(&format_number(*n)),
            Expr::Str(s, _) => self.push(&format!("\"{s}\"")),
            Expr::Ident(name, _) => self.push(name),
            Expr::Binary { op, lhs, rhs, .. } => {
                self.expr(lhs);
                self.push(&format!(" {} ", op_symbol(*op)));
                self.expr(rhs);
            }
            Expr::Call { callee, args, .. } => {
                self.expr(callee);
                self.push("(");
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        self.push(", ");
                    }
                    self.expr(arg);
                }
                self.push(")");
            }
            Expr::Member { obj, field, .. } => {
                self.expr(obj);
                self.push(&format!(".{field}"));
            }
        }
    }
}

// ============================ helpers ================================

fn type_name(ty: &Type) -> String {
    match ty {
        Type::Named(name) => name.clone(),
    }
}

fn op_symbol(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
    }
}

/// 1.0 → "1.0" (mantém o ponto para f64 ser explícito na leitura).
fn format_number(n: f64) -> String {
    if n == n.trunc() && n.is_finite() && n.abs() < 1e15 {
        format!("{:.1}", n)
    } else {
        n.to_string()
    }
}

/// Ponto de entrada: programa → texto canônico.
pub fn format_program(program: &Program) -> String {
    Formatter::format_program(program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Span;
    use crate::parser::parse_program;

    // ---- normalização estrutural (ignora spans) ----

    fn normalize_expr(e: &Expr) -> Expr {
        let s = Span::new(0, 0);
        match e {
            Expr::Number(n, _) => Expr::Number(*n, s),
            Expr::Str(x, _) => Expr::Str(x.clone(), s),
            Expr::Ident(x, _) => Expr::Ident(x.clone(), s),
            Expr::Binary { op, lhs, rhs, .. } => Expr::Binary {
                op: *op,
                lhs: Box::new(normalize_expr(lhs)),
                rhs: Box::new(normalize_expr(rhs)),
                span: s,
            },
            Expr::Call { callee, args, .. } => Expr::Call {
                callee: Box::new(normalize_expr(callee)),
                args: args.iter().map(normalize_expr).collect(),
                span: s,
            },
            Expr::Member { obj, field, .. } => Expr::Member {
                obj: Box::new(normalize_expr(obj)),
                field: field.clone(),
                span: s,
            },
        }
    }

    fn normalize(p: &Program) -> Program {
        let s = Span::new(0, 0);
        Program {
            imports: p
                .imports
                .iter()
                .map(|i| Import { names: i.names.clone(), from: i.from.clone(), span: s })
                .collect(),
            decls: p
                .decls
                .iter()
                .map(|d| match d {
                    Decl::Enum(e) => Decl::Enum(EnumDecl {
                        name: e.name.clone(),
                        variants: e.variants.clone(),
                        span: s,
                    }),
                    Decl::Struct(st) => Decl::Struct(StructDecl {
                        name: st.name.clone(),
                        fields: st
                            .fields
                            .iter()
                            .map(|f| Field { name: f.name.clone(), ty: f.ty.clone(), span: s })
                            .collect(),
                        span: s,
                    }),
                    Decl::Func(f) => Decl::Func(FuncDecl {
                        receiver: f.receiver.clone(),
                        name: f.name.clone(),
                        params: f.params.clone(),
                        ret: f.ret.clone(),
                        body: f
                            .body
                            .iter()
                            .map(|b| match b {
                                Stmt::Let { name, ty, value, .. } => Stmt::Let {
                                    name: name.clone(),
                                    ty: ty.clone(),
                                    value: normalize_expr(value),
                                    span: s,
                                },
                                Stmt::Return(v, _) => Stmt::Return(v.as_ref().map(normalize_expr), s),
                                Stmt::Expr(e) => Stmt::Expr(normalize_expr(e)),
                            })
                            .collect(),
                        span: s,
                    }),
                })
                .collect(),
        }
    }

    #[test]
    fn formats_a_program() {
        let src = r#"import { City } from "engine";
enum Mood { Happy, Neutral }
struct C { x: float; }
func (c: C) go(city: City) { let a: float = 1; c.move(city, a); }"#;
        let program = parse_program(src).unwrap();
        let formatted = format_program(&program);
        let expected = r#"import { City } from "engine";

enum Mood {
    Happy,
    Neutral,
}

struct C {
    x: float;
}

func (c: C) go(city: City) {
    let a: float = 1.0;
    c.move(city, a);
}
"#;
        assert_eq!(formatted, expected);
    }

    #[test]
    fn formatting_is_idempotent() {
        let src = r#"import { City } from "engine";
struct C { x: float; }
func (c: C) go(city: City) { let a = 1; c.move(city, a); }"#;
        let once = format_program(&parse_program(src).unwrap());
        let twice = format_program(&parse_program(&once).unwrap());
        assert_eq!(once, twice);
    }

    #[test]
    fn formatting_roundtrips() {
        // parse(fmt(x)) deve reproduzir a mesma AST que parse(x)
        // (comparação estrutural: spans mudam com a formatação).
        let src = r#"import { City } from "engine";
struct C { x: float; }
func (c: C) go(city: City) { let a = 1; c.move(city, a); }"#;
        let original = normalize(&parse_program(src).unwrap());
        let formatted = format_program(&parse_program(src).unwrap());
        let reparsed = normalize(&parse_program(&formatted).unwrap());
        assert_eq!(original, reparsed);
    }
}
