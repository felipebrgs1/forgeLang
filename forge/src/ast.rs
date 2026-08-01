//! AST: a representação intermediária entre o parser e o codegen.
//!
//! Dois andares:
//!   Program (declarações top-level) → Decl → ... → Stmt → Expr
//!
//! O type checker (F4) vai consumir este mesmo AST e produzir
//! uma versão "tipada" que o codegen entende.

use crate::lexer::Span;

// ============================= Tipos =================================

/// Tipos da linguagem. Hoje: apenas tipos nomeados (float, City, Vec2...).
/// F4 adiciona: genéricos, arrays, optionals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Named(String),
}

// ========================== Declarações ==============================

/// Um programa inteiro: imports + declarações.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub imports: Vec<Import>,
    pub decls: Vec<Decl>,
}

/// `import { City, Vec2 } from "engine"`
#[derive(Debug, Clone, PartialEq)]
pub struct Import {
    pub names: Vec<String>,
    pub from: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decl {
    Enum(EnumDecl),
    Struct(StructDecl),
    Func(FuncDecl),
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumDecl {
    pub name: String,
    pub variants: Vec<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructDecl {
    pub name: String,
    pub fields: Vec<Field>,
    pub span: Span,
}

/// Campo de struct: `name: Type`
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

/// Função ou método (se `receiver` for Some).
/// `func (c: Citizen) update(city: City, dt: float) { ... }`
#[derive(Debug, Clone, PartialEq)]
pub struct FuncDecl {
    pub receiver: Option<Receiver>,
    pub name: String,
    pub params: Vec<Param>,
    /// `None` = sem retorno (void)
    pub ret: Option<Type>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

/// Receiver estilo Go: `(c: Citizen)` — vira o parâmetro 0 no codegen.
#[derive(Debug, Clone, PartialEq)]
pub struct Receiver {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

// =========================== Statements ==============================

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// `let x: float = 1.0;` (tipo opcional — inferência na F4)
    Let {
        name: String,
        ty: Option<Type>,
        value: Expr,
        span: Span,
    },
    /// `return expr;`
    Return(Option<Expr>, Span),
    /// Qualquer expressão como statement (chamadas, etc.)
    Expr(Expr),
}

// ============================ Expressões ==============================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Number(f64, Span),
    Str(String, Span),
    Ident(String, Span),
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },
    /// Chamada de função: `f(a, b)` — o callee pode ser um member: `c.update(...)`
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },
    /// Acesso a membro: `obj.field`
    Member {
        obj: Box<Expr>,
        field: String,
        span: Span,
    },
}

// `Expr::span` será usado pelo codegen/erros nas próximas fases.
#[allow(dead_code)]
impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Number(_, s)
            | Expr::Str(_, s)
            | Expr::Ident(_, s) => *s,
            Expr::Binary { span, .. }
            | Expr::Call { span, .. }
            | Expr::Member { span, .. } => *span,
        }
    }
}
