//! Lexer: texto-fonte → tokens, com spans para mensagens de erro.

/// Palavras-chave da linguagem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyword {
    Let,
    Func,
    Struct,
    Enum,
    Import,
    From,
    Return,
}

impl Keyword {
    fn from_str(s: &str) -> Option<Keyword> {
        Some(match s {
            "let" => Keyword::Let,
            "func" => Keyword::Func,
            "struct" => Keyword::Struct,
            "enum" => Keyword::Enum,
            "import" => Keyword::Import,
            "from" => Keyword::From,
            "return" => Keyword::Return,
            _ => return None,
        })
    }
}

/// Posição no código-fonte (1-indexed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub line: u32,
    pub col: u32,
}

impl Span {
    pub fn new(line: u32, col: u32) -> Self {
        Self { line, col }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Number(f64),
    /// Identificador (inclui `self`, que é convenção, não keyword).
    Ident(String),
    /// String literal: "engine"
    StrLit(String),
    Keyword(Keyword),
    Plus,
    Minus,
    Star,
    Slash,
    Eq,
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Semicolon,
    Dot,
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }
}

#[derive(Debug)]
pub struct LexError {
    pub msg: String,
    pub span: Span,
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: erro de lexer: {}", self.span.line, self.span.col, self.msg)
    }
}

pub fn lex(src: &str) -> Result<Vec<Token>, LexError> {
    let chars: Vec<char> = src.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0usize;
    let mut line = 1u32;
    let mut col = 1u32;

    // push com avanço automático de posição.
    macro_rules! push {
        ($kind:expr, $span:expr) => {{
            tokens.push(Token::new($kind, $span));
        }};
    }

    while i < chars.len() {
        let c = chars[i];
        let span = Span::new(line, col);

        match c {
            // Whitespace.
            ' ' | '\t' | '\r' => {
                i += 1;
                col += 1;
            }
            '\n' => {
                i += 1;
                line += 1;
                col = 1;
            }
            // Comentário de linha: // até o fim da linha.
            '/' if i + 1 < chars.len() && chars[i + 1] == '/' => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                    col += 1;
                }
            }
            // Operadores e pontuação.
            '+' => {
                push!(TokenKind::Plus, span);
                i += 1;
                col += 1;
            }
            '-' => {
                push!(TokenKind::Minus, span);
                i += 1;
                col += 1;
            }
            '*' => {
                push!(TokenKind::Star, span);
                i += 1;
                col += 1;
            }
            '/' => {
                push!(TokenKind::Slash, span);
                i += 1;
                col += 1;
            }
            '=' => {
                push!(TokenKind::Eq, span);
                i += 1;
                col += 1;
            }
            '(' => {
                push!(TokenKind::LParen, span);
                i += 1;
                col += 1;
            }
            ')' => {
                push!(TokenKind::RParen, span);
                i += 1;
                col += 1;
            }
            '{' => {
                push!(TokenKind::LBrace, span);
                i += 1;
                col += 1;
            }
            '}' => {
                push!(TokenKind::RBrace, span);
                i += 1;
                col += 1;
            }
            ',' => {
                push!(TokenKind::Comma, span);
                i += 1;
                col += 1;
            }
            ':' => {
                push!(TokenKind::Colon, span);
                i += 1;
                col += 1;
            }
            ';' => {
                push!(TokenKind::Semicolon, span);
                i += 1;
                col += 1;
            }
            // Número: "42", "3.14", ".5" — mas "." sozinho é member access.
            '0'..='9' | '.'
                if c.is_ascii_digit()
                    || chars.get(i + 1).is_some_and(|n| n.is_ascii_digit()) =>
            {
                let start = i;
                let mut seen_dot = false;
                while i < chars.len() {
                    match chars[i] {
                        '0'..='9' => {}
                        '.' if !seen_dot => seen_dot = true,
                        _ => break,
                    }
                    i += 1;
                    col += 1;
                }
                let text: String = chars[start..i].iter().collect();
                match text.parse::<f64>() {
                    Ok(n) => push!(TokenKind::Number(n), span),
                    Err(_) => {
                        return Err(LexError {
                            msg: format!("numero invalido: '{text}'"),
                            span,
                        })
                    }
                }
            }
            '.' => {
                push!(TokenKind::Dot, span);
                i += 1;
                col += 1;
            }
            // Identificador ou keyword.
            'a'..='z' | 'A'..='Z' | '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                    col += 1;
                }
                let text: String = chars[start..i].iter().collect();
                match Keyword::from_str(&text) {
                    Some(kw) => push!(TokenKind::Keyword(kw), span),
                    None => push!(TokenKind::Ident(text), span),
                }
            }
            // String: "engine"
            '"' => {
                i += 1;
                col += 1;
                let start = i;
                while i < chars.len() && chars[i] != '"' && chars[i] != '\n' {
                    i += 1;
                    col += 1;
                }
                if i >= chars.len() || chars[i] != '"' {
                    return Err(LexError {
                        msg: "string nao fechada".into(),
                        span,
                    });
                }
                let text: String = chars[start..i].iter().collect();
                push!(TokenKind::StrLit(text), span);
                i += 1;
                col += 1;
            }
            other => {
                return Err(LexError {
                    msg: format!("caractere inesperado: '{other}'"),
                    span,
                })
            }
        }
    }

    tokens.push(Token::new(TokenKind::Eof, Span::new(line, col)));
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lex_simple_expression() {
        let tokens = lex("1 + 2 * (3 - 4)").unwrap();
        let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind.clone()).collect();
        assert_eq!(
            kinds,
            vec![
                TokenKind::Number(1.0),
                TokenKind::Plus,
                TokenKind::Number(2.0),
                TokenKind::Star,
                TokenKind::LParen,
                TokenKind::Number(3.0),
                TokenKind::Minus,
                TokenKind::Number(4.0),
                TokenKind::RParen,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn lex_keywords_and_idents() {
        let tokens = lex("let x = 1; func (c: City) update() {}").unwrap();
        assert!(matches!(tokens[0].kind, TokenKind::Keyword(Keyword::Let)));
        assert!(matches!(tokens[1].kind, TokenKind::Ident(ref s) if s == "x"));
        assert!(matches!(tokens[5].kind, TokenKind::Keyword(Keyword::Func)));
        // `City` é ident, `self` também (convenção, não keyword).
        assert!(matches!(tokens[9].kind, TokenKind::Ident(ref s) if s == "City"));
    }

    #[test]
    fn lex_string_literal() {
        let tokens = lex("import { A } from \"engine\"").unwrap();
        assert!(matches!(tokens[5].kind, TokenKind::StrLit(ref s) if s == "engine"));
    }

    #[test]
    fn lex_skips_line_comments() {
        let tokens = lex("1 + // comentario\n2").unwrap();
        let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind.clone()).collect();
        assert_eq!(kinds, vec![TokenKind::Number(1.0), TokenKind::Plus, TokenKind::Number(2.0), TokenKind::Eof]);
    }

    #[test]
    fn lex_tracks_lines() {
        let tokens = lex("1 +\n2").unwrap();
        assert_eq!(tokens[0].span.line, 1);
        assert_eq!(tokens[2].span.line, 2);
    }

    #[test]
    fn lex_rejects_unclosed_string() {
        assert!(lex("\"engine").is_err());
    }
}
