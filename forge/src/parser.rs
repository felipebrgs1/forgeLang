//! Parser: tokens → AST.
//!
//! Dois andares:
//!   1. Declarações top-level (import/enum/struct/func) — decididas por keyword
//!   2. Expressões — precedence climbing (additive → multiplicative → postfix → primary)
//!
//! Receiver estilo Go: `func (c: Citizen) update(...)` é parseado como
//! um "parâmetro 0 disfarçado" — o codegen (F2) vai tratá-lo assim.

use crate::ast::*;
use crate::lexer::{lex, Keyword, Span, Token, TokenKind};

#[derive(Debug)]
pub struct ParseError {
    pub msg: String,
    pub span: Span,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: erro de parser: {}", self.span.line, self.span.col, self.msg)
    }
}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    // --------------------- helpers de token -------------------------

    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn peek_kind(&self) -> &TokenKind {
        &self.peek().kind
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos];
        if tok.kind != TokenKind::Eof {
            self.pos += 1;
        }
        &self.tokens[self.pos - 1]
    }

    fn error(&self, msg: impl Into<String>) -> ParseError {
        ParseError {
            msg: msg.into(),
            span: self.peek().span,
        }
    }

    fn expect(&mut self, kind: &TokenKind) -> Result<&Token, ParseError> {
        if self.peek_kind() == kind {
            Ok(self.advance())
        } else {
            Err(self.error(format!("esperava {:?}, encontrou {:?}", kind, self.peek_kind())))
        }
    }

    fn expect_ident(&mut self) -> Result<(String, Span), ParseError> {
        let kind = self.peek_kind().clone();
        match kind {
            TokenKind::Ident(name) => {
                let span = self.advance().span;
                Ok((name, span))
            }
            _ => Err(self.error(format!(
                "esperava identificador, encontrou {:?}",
                self.peek_kind()
            ))),
        }
    }

    fn expect_keyword(&mut self, kw: Keyword) -> Result<Span, ParseError> {
        match self.peek_kind() {
            TokenKind::Keyword(k) if *k == kw => {
                let span = self.advance().span;
                Ok(span)
            }
            _ => Err(self.error(format!("esperava keyword {:?}", kw))),
        }
    }

    fn at_keyword(&self, kw: Keyword) -> bool {
        matches!(self.peek_kind(), TokenKind::Keyword(k) if *k == kw)
    }

    // -------------------- andar 1: declarações ------------------------

    /// Ponto de entrada: parseia o programa inteiro.
    pub fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut imports = Vec::new();
        let mut decls = Vec::new();

        while self.peek_kind() != &TokenKind::Eof {
            if self.at_keyword(Keyword::Import) {
                imports.push(self.parse_import()?);
            } else if self.at_keyword(Keyword::Enum) {
                decls.push(Decl::Enum(self.parse_enum()?));
            } else if self.at_keyword(Keyword::Struct) {
                decls.push(Decl::Struct(self.parse_struct()?));
            } else if self.at_keyword(Keyword::Func) {
                decls.push(Decl::Func(self.parse_func()?));
            } else {
                return Err(self.error(format!(
                    "esperava declaração (import/enum/struct/func), encontrou {:?}",
                    self.peek_kind()
                )));
            }
        }
        Ok(Program { imports, decls })
    }

    /// `import { A, B } from "engine";`
    fn parse_import(&mut self) -> Result<Import, ParseError> {
        let span = self.expect_keyword(Keyword::Import)?;
        self.expect(&TokenKind::LBrace)?;

        let mut names = Vec::new();
        loop {
            let (name, _) = self.expect_ident()?;
            names.push(name);
            if self.peek_kind() == &TokenKind::Comma {
                self.advance();
            } else {
                break;
            }
        }
        self.expect(&TokenKind::RBrace)?;
        self.expect_keyword(Keyword::From)?;

        let from = match self.peek_kind() {
            TokenKind::StrLit(s) => {
                let s = s.clone();
                self.advance();
                s
            }
            _ => return Err(self.error("esperava string com o caminho do import")),
        };
        self.expect(&TokenKind::Semicolon)?;
        Ok(Import { names, from, span })
    }

    /// `enum Mood { Happy, Neutral, Stressed }`
    fn parse_enum(&mut self) -> Result<EnumDecl, ParseError> {
        let span = self.expect_keyword(Keyword::Enum)?;
        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::LBrace)?;

        let mut variants = Vec::new();
        loop {
            let (variant, _) = self.expect_ident()?;
            variants.push(variant);
            if self.peek_kind() == &TokenKind::Comma {
                self.advance();
                if self.peek_kind() == &TokenKind::RBrace {
                    break; // vírgula final
                }
            } else {
                break;
            }
        }
        self.expect(&TokenKind::RBrace)?;
        Ok(EnumDecl { name, variants, span })
    }

    /// `struct Citizen { home: Vec2; work: Vec2; }`
    fn parse_struct(&mut self) -> Result<StructDecl, ParseError> {
        let span = self.expect_keyword(Keyword::Struct)?;
        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::LBrace)?;

        let mut fields = Vec::new();
        while self.peek_kind() != &TokenKind::RBrace {
            fields.push(self.parse_field()?);
            self.expect(&TokenKind::Semicolon)?;
        }
        self.expect(&TokenKind::RBrace)?;
        Ok(StructDecl { name, fields, span })
    }

    fn parse_field(&mut self) -> Result<Field, ParseError> {
        let (name, span) = self.expect_ident()?;
        self.expect(&TokenKind::Colon)?;
        let ty = self.parse_type()?;
        Ok(Field { name, ty, span })
    }

    /// `func (c: Citizen) update(city: City, dt: float) { ... }`
    fn parse_func(&mut self) -> Result<FuncDecl, ParseError> {
        let span = self.expect_keyword(Keyword::Func)?;

        // Receiver estilo Go: func (name: Type) name(...)
        let receiver = if self.peek_kind() == &TokenKind::LParen {
            self.advance();
            let (name, _) = self.expect_ident()?;
            self.expect(&TokenKind::Colon)?;
            let ty = self.parse_type()?;
            self.expect(&TokenKind::RParen)?;
            Some(Receiver { name, ty })
        } else {
            None
        };

        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::LParen)?;

        let mut params = Vec::new();
        if self.peek_kind() != &TokenKind::RParen {
            loop {
                let (pname, _) = self.expect_ident()?;
                self.expect(&TokenKind::Colon)?;
                let pty = self.parse_type()?;
                params.push(Param { name: pname, ty: pty });
                if self.peek_kind() == &TokenKind::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(&TokenKind::RParen)?;

        // Retorno opcional no estilo TS: func f(...) : float
        let ret = if self.peek_kind() == &TokenKind::Colon {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = self.parse_block()?;
        Ok(FuncDecl { receiver, name, params, ret, body, span })
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, ParseError> {
        self.expect(&TokenKind::LBrace)?;
        let mut stmts = Vec::new();
        while self.peek_kind() != &TokenKind::RBrace {
            stmts.push(self.parse_stmt()?);
        }
        self.expect(&TokenKind::RBrace)?;
        Ok(stmts)
    }

    // --------------------- andar 2: statements ------------------------

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        if self.at_keyword(Keyword::Let) {
            self.parse_let()
        } else if self.at_keyword(Keyword::Return) {
            self.parse_return()
        } else if self.at_keyword(Keyword::If) {
            self.parse_if()
        } else if self.at_keyword(Keyword::For) {
            self.parse_for()
        } else if self.at_keyword(Keyword::Break) {
            let span = self.advance().span;
            self.expect(&TokenKind::Semicolon)?;
            Ok(Stmt::Break(span))
        } else if self.at_keyword(Keyword::Continue) {
            let span = self.advance().span;
            self.expect(&TokenKind::Semicolon)?;
            Ok(Stmt::Continue(span))
        } else {
            let expr = self.parse_expr()?;
            if self.peek_kind() == &TokenKind::Eq {
                // atribuição: expr '=' expr ';'
                let span = self.advance().span;
                let value = self.parse_expr()?;
                self.expect(&TokenKind::Semicolon)?;
                Ok(Stmt::Assign { target: expr, value, span })
            } else {
                self.expect(&TokenKind::Semicolon)?;
                Ok(Stmt::Expr(expr))
            }
        }
    }

    /// `if cond { } else { }` — `else if` vira else_body = [If].
    fn parse_if(&mut self) -> Result<Stmt, ParseError> {
        let span = self.expect_keyword(Keyword::If)?;
        let cond = self.parse_expr()?;
        let then_body = self.parse_block()?;
        let else_body = if self.at_keyword(Keyword::Else) {
            self.advance();
            if self.at_keyword(Keyword::If) {
                Some(vec![self.parse_if()?])
            } else {
                Some(self.parse_block()?)
            }
        } else {
            None
        };
        Ok(Stmt::If { cond, then_body, else_body, span })
    }

    /// `for` unificado. As formas são distinguidas por lookahead:
    ///   for { }                 → infinito
    ///   for let i = 0.0; ...    → contador (init statement)
    ///   for i = 0.0; ...        → contador (init expr)
    ///   for cond { }            → enquanto
    ///   for x in xs { }         → iteração (parse; codegen exige arrays: F5)
    fn parse_for(&mut self) -> Result<Stmt, ParseError> {
        let span = self.expect_keyword(Keyword::For)?;

        // for { } — infinito
        if self.peek_kind() == &TokenKind::LBrace {
            let body = self.parse_block()?;
            return Ok(Stmt::For { init: None, cond: None, post: None, body, span });
        }

        // for x in xs { } — iteração (detectada por Ident + keyword `in`)
        if matches!(self.peek_kind(), TokenKind::Ident(_))
            && matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::Keyword(Keyword::In)))
        {
            self.advance(); // ident
            self.advance(); // in
            let _iterable = self.parse_expr()?;
            let _body = self.parse_block()?;
            return Err(ParseError {
                msg: "iteração 'for x in xs' requer arrays — chega na F5".into(),
                span,
            });
        }

        // for let i = 0.0; cond; post { } — contador com init let
        if self.at_keyword(Keyword::Let) {
            let init = self.parse_let()?;
            let cond = if self.peek_kind() == &TokenKind::Semicolon {
                self.advance();
                None
            } else {
                let c = self.parse_expr()?;
                self.expect(&TokenKind::Semicolon)?;
                Some(c)
            };
            let post = self.parse_for_post()?;
            let body = self.parse_block()?;
            return Ok(Stmt::For {
                init: Some(Box::new(init)),
                cond,
                post,
                body,
                span,
            });
        }

        // Decide a forma pelo delimitador após a primeira expressão:
        //   expr { }  → for cond (enquanto)
        //   expr ;    → contador (init)
        let first = self.parse_expr()?;
        if self.peek_kind() == &TokenKind::LBrace {
            let body = self.parse_block()?;
            return Ok(Stmt::For {
                init: None,
                cond: Some(first),
                post: None,
                body,
                span,
            });
        }
        self.expect(&TokenKind::Semicolon)?;

        // cond: expr, terminado por ';' (vazio = sempre verdadeiro)
        let cond = if self.peek_kind() == &TokenKind::Semicolon {
            self.advance();
            None
        } else {
            let c = self.parse_expr()?;
            self.expect(&TokenKind::Semicolon)?;
            Some(c)
        };

        let post = self.parse_for_post()?;
        let body = self.parse_block()?;
        Ok(Stmt::For {
            init: Some(Box::new(Stmt::Expr(first))),
            cond,
            post,
            body,
            span,
        })
    }

    /// Post de um for contador: statement sem `;` final
    /// (geralmente atribuição) ou nada se `{` já chegou.
    fn parse_for_post(&mut self) -> Result<Option<Box<Stmt>>, ParseError> {
        if self.peek_kind() == &TokenKind::LBrace {
            return Ok(None);
        }
        if self.at_keyword(Keyword::Let) {
            return Ok(Some(Box::new(self.parse_let()?)));
        }
        let e = self.parse_expr()?;
        if self.peek_kind() == &TokenKind::Eq {
            let span = self.advance().span;
            let v = self.parse_expr()?;
            Ok(Some(Box::new(Stmt::Assign { target: e, value: v, span })))
        } else {
            Ok(Some(Box::new(Stmt::Expr(e))))
        }
    }

    /// `let x: float = 1.0;`
    fn parse_let(&mut self) -> Result<Stmt, ParseError> {
        let span = self.expect_keyword(Keyword::Let)?;
        let (name, _) = self.expect_ident()?;

        let ty = if self.peek_kind() == &TokenKind::Colon {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(&TokenKind::Eq)?;
        let value = self.parse_expr()?;
        self.expect(&TokenKind::Semicolon)?;
        Ok(Stmt::Let { name, ty, value, span })
    }

    /// `return expr;` ou `return;`
    fn parse_return(&mut self) -> Result<Stmt, ParseError> {
        let span = self.expect_keyword(Keyword::Return)?;
        let value = if self.peek_kind() == &TokenKind::Semicolon {
            None
        } else {
            Some(self.parse_expr()?)
        };
        self.expect(&TokenKind::Semicolon)?;
        Ok(Stmt::Return(value, span))
    }

    // --------------------- tipos --------------------------------------

    fn parse_type(&mut self) -> Result<Type, ParseError> {
        let (name, _) = self.expect_ident()?;
        Ok(Type::Named(name))
    }

    // --------------------- expressões ---------------------------------

    /// Expressão no topo (usada pelo CLI/JIT e por statements).
    pub fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_and()?;
        while self.peek_kind() == &TokenKind::OrOr {
            let span = self.advance().span;
            let rhs = self.parse_and()?;
            lhs = Expr::Binary { op: BinOp::Or, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_equality()?;
        while self.peek_kind() == &TokenKind::AndAnd {
            let span = self.advance().span;
            let rhs = self.parse_equality()?;
            lhs = Expr::Binary { op: BinOp::And, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }

    fn parse_equality(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_relational()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::EqEq => BinOp::Eq,
                TokenKind::Ne => BinOp::Ne,
                _ => break,
            };
            let span = self.advance().span;
            let rhs = self.parse_relational()?;
            lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }

    fn parse_relational(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_additive()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Lt => BinOp::Lt,
                TokenKind::Le => BinOp::Le,
                TokenKind::Gt => BinOp::Gt,
                TokenKind::Ge => BinOp::Ge,
                _ => break,
            };
            let span = self.advance().span;
            let rhs = self.parse_additive()?;
            lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }

    fn parse_additive(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_multiplicative()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Plus => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                _ => break,
            };
            let span = self.advance().span;
            let rhs = self.parse_multiplicative()?;
            lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Star => BinOp::Mul,
                TokenKind::Slash => BinOp::Div,
                _ => break,
            };
            let span = self.advance().span;
            let rhs = self.parse_unary()?;
            lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }

    /// Unário: `-x`, `!x`, encadeável (`--x` para números negativos).
    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        let tok = self.peek().clone();
        let op = match tok.kind {
            TokenKind::Minus => UnOp::Neg,
            TokenKind::Bang => UnOp::Not,
            _ => return self.parse_postfix(),
        };
        self.advance();
        let operand = self.parse_unary()?;
        Ok(Expr::Unary { op, operand: Box::new(operand), span: tok.span })
    }

    /// Pós-fixo: chamadas e acesso a membro ligam mais forte que binários.
    /// `c.update(city, dt)` → Call(Member(c, "update"), [city, dt])
    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek_kind() {
                TokenKind::LParen => {
                    let span = self.advance().span;
                    let mut args = Vec::new();
                    if self.peek_kind() != &TokenKind::RParen {
                        loop {
                            args.push(self.parse_or()?);
                            if self.peek_kind() == &TokenKind::Comma {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(&TokenKind::RParen)?;
                    expr = Expr::Call { callee: Box::new(expr), args, span };
                }
                TokenKind::Dot => {
                    let span = self.advance().span;
                    let (field, _) = self.expect_ident()?;
                    expr = Expr::Member { obj: Box::new(expr), field, span };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let tok = self.peek().clone();
        match tok.kind {
            TokenKind::Number(n) => {
                self.advance();
                Ok(Expr::Number(n, tok.span))
            }
            TokenKind::StrLit(s) => {
                self.advance();
                Ok(Expr::Str(s, tok.span))
            }
            TokenKind::Ident(name) => {
                self.advance();
                // `Point { x: 1.0 }` — literal de struct. Só quando o
                // padrão Ident { Ident : — senão `a {` é bloco de if/for.
                let is_struct_lit = matches!(self.peek_kind(), TokenKind::LBrace)
                    && matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::Ident(_)))
                    && matches!(self.tokens.get(self.pos + 2).map(|t| &t.kind), Some(TokenKind::Colon));
                if is_struct_lit {
                    self.advance(); // {
                    let mut fields = Vec::new();
                    while self.peek_kind() != &TokenKind::RBrace {
                        let (fname, _) = self.expect_ident()?;
                        self.expect(&TokenKind::Colon)?;
                        let value = self.parse_or()?;
                        fields.push((fname, value));
                        if self.peek_kind() == &TokenKind::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    self.expect(&TokenKind::RBrace)?;
                    Ok(Expr::StructLit { name, fields, span: tok.span })
                } else {
                    Ok(Expr::Ident(name, tok.span))
                }
            }
            TokenKind::LParen => {
                self.advance();
                let inner = self.parse_or()?;
                self.expect(&TokenKind::RParen)?;
                Ok(inner)
            }
            _ => Err(self.error(format!("expressão inválida, encontrou {:?}", tok.kind))),
        }
    }
}

// ========================== Conveniência =============================

/// Texto → AST de programa (declarações).
pub fn parse_program(src: &str) -> Result<Program, ParseError> {
    let tokens = lex(src).map_err(|e| ParseError { msg: e.msg, span: e.span })?;
    Parser::new(tokens).parse_program()
}

/// Texto → AST de expressão (caminho F1, usado pelo JIT).
pub fn parse_expr(src: &str) -> Result<Expr, ParseError> {
    let tokens = lex(src).map_err(|e| ParseError { msg: e.msg, span: e.span })?;
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expr()?;
    parser.expect(&TokenKind::Eof)?;
    Ok(expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ops_in(expr: &Expr, out: &mut Vec<BinOp>) {
        match expr {
            Expr::Binary { op, lhs, rhs, .. } => {
                out.push(*op);
                ops_in(lhs, out);
                ops_in(rhs, out);
            }
            Expr::Call { callee, args, .. } => {
                ops_in(callee, out);
                for a in args {
                    ops_in(a, out);
                }
            }
            Expr::Member { obj, .. } => ops_in(obj, out),
            Expr::StructLit { fields, .. } => {
                for (_, e) in fields {
                    ops_in(e, out);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn precedence_mul_over_add() {
        let expr = parse_expr("1 + 2 * 3").unwrap();
        let Expr::Binary { op: BinOp::Add, rhs, .. } = expr else {
            panic!("esperava Add no topo");
        };
        assert!(matches!(*rhs, Expr::Binary { op: BinOp::Mul, .. }));
    }

    #[test]
    fn parens_override_precedence() {
        let expr = parse_expr("(1 + 2) * 3").unwrap();
        let Expr::Binary { op: BinOp::Mul, lhs, .. } = expr else {
            panic!("esperava Mul no topo");
        };
        assert!(matches!(*lhs, Expr::Binary { op: BinOp::Add, .. }));
    }

    #[test]
    fn associativity_is_left() {
        let expr = parse_expr("10 - 3 - 2").unwrap();
        let mut ops = Vec::new();
        ops_in(&expr, &mut ops);
        assert_eq!(ops, vec![BinOp::Sub, BinOp::Sub]);
    }

    #[test]
    fn postfix_call_and_member() {
        // c.update(city, dt) → Call(Member(c, update), [city, dt])
        let expr = parse_expr("c.update(city, dt)").unwrap();
        let Expr::Call { callee, args, .. } = expr else {
            panic!("esperava Call");
        };
        assert!(matches!(*callee, Expr::Member { ref field, .. } if field == "update"));
        assert_eq!(args.len(), 2);
    }

    #[test]
    fn parse_program_with_all_decls() {
        let src = r#"
            import { City, Vec2 } from "engine";

            enum Mood { Happy, Neutral }

            struct Citizen {
                home: Vec2;
                mood: Mood;
            }

            func (c: Citizen) update(city: City, dt: float) {
                let speed: float = c.home.distance(city.center);
                return;
            }
        "#;
        let program = parse_program(src).unwrap();
        assert_eq!(program.imports.len(), 1);
        assert_eq!(program.decls.len(), 3);

        let Decl::Struct(s) = &program.decls[1] else { panic!() };
        assert_eq!(s.name, "Citizen");
        assert_eq!(s.fields.len(), 2);

        let Decl::Func(f) = &program.decls[2] else { panic!() };
        assert!(f.receiver.is_some());
        assert_eq!(f.params.len(), 2);
        assert_eq!(f.body.len(), 2);
    }

    #[test]
    fn rejects_bad_declaration() {
        assert!(parse_program("banana x = 1").is_err());
    }
}
