//! AST: a representação intermediária entre o parser e o codegen.
//!
//! Dois andares:
//!   Program (declarações top-level) → Decl → ... → Stmt → Expr
//!
//! O type checker (F4) consome este AST e produz uma versão "tipada":
//! a mesma árvore com `Cast` inseridos para coerção int→float.

use crate::lexer::Span;

// ============================= Tipos =================================

/// Tipos da linguagem.
///
/// F4: primitivos `int`/`float`/`bool`/`string`/`void` + tipos nomeados
/// (structs e enums). F5 adiciona: genéricos, arrays, optionals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    /// Inteiro com sinal (i64 internamente). `42` — divisão trunca (Go).
    Int,
    /// Ponto flutuante (f64). `42.0`, `3.14`.
    Float,
    /// `true`/`false` (i1 internamente).
    Bool,
    /// String literal internada (ponteiro para global única por conteúdo).
    Str,
    /// Sem valor — só em posição de retorno de função.
    Void,
    /// Struct ou enum declarado no programa.
    Named(String),
}

/// Nome canônico de um tipo para mensagens de erro e formatação.
pub fn type_name(ty: &Type) -> String {
    match ty {
        Type::Int => "int".into(),
        Type::Float => "float".into(),
        Type::Bool => "bool".into(),
        Type::Str => "string".into(),
        Type::Void => "void".into(),
        Type::Named(name) => name.clone(),
    }
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
    /// Atribuição: `alvo = valor;` (alvo é lvalue: variável ou campo)
    Assign {
        target: Expr,
        value: Expr,
        span: Span,
    },
    /// `if cond { ... } else { ... }` (else é opcional; `else if` vira
    /// else_body = [If {...}]).
    If {
        cond: Expr,
        then_body: Vec<Stmt>,
        else_body: Option<Vec<Stmt>>,
        span: Span,
    },
    /// `for` unificado (Go):
    ///   for {}                     — infinito
    ///   for cond {}                — enquanto
    ///   for init; cond; post {}    — contador
    ///   for x in xs {}             — iteração (requer arrays: F5)
    For {
        init: Option<Box<Stmt>>,
        cond: Option<Expr>,
        post: Option<Box<Stmt>>,
        body: Vec<Stmt>,
        span: Span,
    },
    /// `break;` — sai do loop mais interno.
    Break(Span),
    /// `continue;` — pula para a próxima iteração.
    Continue(Span),
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
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    /// `-x`
    Neg,
    /// `!x`
    Not,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Inteiro literal: `42`.
    Int(i64, Span),
    /// Float literal: `42.0`, `3.14`.
    Float(f64, Span),
    /// `true` / `false`.
    Bool(bool, Span),
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
    /// Literal de struct: `Point { x: 1.0, y: 2.0 }`
    StructLit {
        name: String,
        fields: Vec<(String, Expr)>,
        span: Span,
    },
    /// Operação unária: `-x`, `!x`
    Unary {
        op: UnOp,
        operand: Box<Expr>,
        span: Span,
    },
    /// Cast explícito (`int(x)`, `float(x)`) — inserido pelo type checker
    /// para coerção int→float implícita, ou escrito pelo usuário para
    /// float→int (o único caminho, como em Go).
    Cast {
        to: Type,
        expr: Box<Expr>,
        span: Span,
    },
}

// `Expr::span` será usado pelo codegen/erros nas próximas fases.
#[allow(dead_code)]
impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Int(_, s)
            | Expr::Float(_, s)
            | Expr::Bool(_, s)
            | Expr::Str(_, s)
            | Expr::Ident(_, s) => *s,
            Expr::Binary { span, .. }
            | Expr::Call { span, .. }
            | Expr::Member { span, .. }
            | Expr::StructLit { span, .. }
            | Expr::Unary { span, .. }
            | Expr::Cast { span, .. } => *span,
        }
    }
}
