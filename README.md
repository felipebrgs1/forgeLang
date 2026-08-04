# engine

Workspace de uma game engine em Rust, começando pela linguagem de script.

## Stack

- **Rust** 1.97
- **LLVM** 22.1.8 (via crate `inkwell` 0.9)
- Linguagem compilada via LLVM, executada por **JIT** — sem interpretador

## forge — a linguagem de script

Linguagem tipo TypeScript (chaves, tipada) focada no ferramental do jogo.
Nome provisório.

Pipeline: `texto → lexer → parser → AST → type checker → codegen (LLVM IR) → JIT → execução`

```
cargo run -p forge -- "1 + 2 * 3"               # → 7
cargo run -p forge -- run <arquivo.forge>       # executa func main() via JIT
cargo run -p forge -- fmt <arquivo.forge>       # formata (estilo canônico)
cargo run -p forge -- ast <arquivo.forge>       # dump da AST
cargo test -p forge                              # 100 testes
```

### Sintaxe (decisão A: receiver Go, aparência TS)

```ts
import { City, Vec2 } from "engine";

enum Mood { Happy, Neutral, Stressed }

struct Citizen {
    home: Vec2;
    work: Vec2;
    mood: Mood;
}

func (c: Citizen) go_home(city: City, dt: float) {  // receiver estilo Go
    let route = city.find_path(c.home, c.work);
    let first: Vec2 = route.first();
    return;
}
```

Regras: statements terminam em `;`, chaves em K&R, indentação 4 espaços.
O formatter é o dono do estilo (filosofia gofmt) e é idempotente
(`fmt(fmt(x)) == fmt(x)` — testado).

### Estrutura

```
forge/src/
├── main.rs       # CLI (expr | run | fmt | ast)
├── lexer.rs      # texto → tokens (com spans para erros)
├── ast.rs        # Program → Decl → Stmt → Expr
├── parser.rs     # 2 andares: declarações (keyword) + expressões (precedência)
├── checker.rs    # type checker: valida e insere casts int→float (F4)
├── formatter.rs  # AST → texto canônico (gofmt-like, idempotente)
└── codegen.rs    # AST verificado → LLVM IR + execução JIT
```

### Roadmap da linguagem

Cada fase termina com algo testável. Nunca avance sem testes verdes.

- [x] **F1 — Aritmética via JIT**: números f64, `+ - * /`, parênteses, precedência
- [x] **F2a — Declarações + formatter**: imports, enums, structs, funcs com receiver
      (Go), `let`/`return`, chamadas/members, `forge fmt`/`forge ast`
- [x] **F2b — Funções e variáveis executáveis**: codegen de `let` (Alloca/Load/Store),
      funções com forward references, chamadas, `return`, `main` como entry
      (`forge run <arquivo>` executa via JIT)
- [x] **F2c — Structs, receiver e member access**: tipos LLVM agregados + GEP,
      métodos (`Point.len_sq` → receiver = param 0), `a.b()` resolvendo pelo tipo,
      struct literals `Point { x: 1.0, y: 2.0 }`, structs por valor
- [x] **F2d — Structs aninhadas, enums e atribuição**: campos de struct em struct
      (GEP encadeado), enums como valores (`Mood.Stressed`), atribuição com
      lvalue (`c.pos.x = ...`), receiver por REFERÊNCIA (métodos mutam o
      original), detecção de ciclo de struct; `citizen.forge` roda de ponta
      a ponta
- [x] **F3 — Controle de fluxo**: `if/else` (incl. `else if` em cadeia), `for` unificado
      (Go — `for`, `for cond`, `for init; cond; post`, sem `while`), `break`/`continue`,
      comparações (`== != < <= > >=`), lógicos (`&& || !`), unário `-`,
      tipo `bool` (i1), loops aninhados;
      `examples/simulacao.forge` — população que decide mood e conta estressados
- [x] **F4 — Type checker + strings**: primitivos `int`/`float`/`bool`/`string`/`void`,
      literais `true`/`false`, int (i64) e float (f64) separados (Go), divisão de
      int trunca (sdiv), coerção assimétrica (int→float automática com `Cast`
      inserido pelo checker; float→int só via `int(x)`), strings internadas
      (global única por conteúdo — `==`/`!=` vira comparação de ponteiro),
      escopos por bloco, aridade de chamadas, enums como i64;
      `checker.rs` é o dono da verdade de tipos — o codegen consome AST já
      verificado (100 testes; 32 só de checker)
- [ ] **F5 — Structs, arrays e memória**: decisão de GC aqui —
      começar com arena por frame, migrar para MMTk se necessário
- [ ] **F6 — Interop com Rust**: ABI `extern "C"`, o script chama a engine
      (a fronteira `ScriptHost` que discutimos)
- [ ] **F7 — Ferramentas**: REPL, hot reload, debug info (LLVM), CLI de arquivos
- [ ] **F8 — VSCode**: split `forge-core`/`forge-cli` desde já (lib pura p/ LSP e
      engine consumirem); TextMate grammar p/ highlight já no F3;
      `forge-lsp` (tower-lsp) com diagnostics + format quando o type checker
      existir; DAP/debugger só com a engine

### Decisões de design (registradas)

1. **Hot path sempre nativo** — script orquestra, nunca implementa algoritmo crítico.
2. **Fronteira agnóstica** — a API script↔engine será uma trait trocável
   (Luau hoje se precisar, WASM amanhã se o JIT não bastar).
3. **Memória**: sem GC embutido no MVP → arena por tick/frame.
   LLVM não dá GC e Rust não tem — esta é a decisão mais cara do projeto.
4. **Determinismo** para replay/multiplayer: floats e seeds controlados.
5. **Um loop só, estilo Go**: `for` unificado (`for`, `for cond`,
   `for init; cond; post`, `for x in xs`) — `while` é açúcar redundante.
6. **Sem tipos de largura fixa na linguagem**: o script não aloca memória
   de simulação (a engine aloca). Layout fino (SoA, `repr(u8)`, bit-packing)
   é decisão do hot path Rust. Tipos finos só numa futura feature de
   data layout p/ save/network, não como tipos universais de gameplay.

## Próximo passo

F5: arrays + memória (decisão de GC: arena por frame → MMTk se precisar),
`for x in xs`, curto-circuito em `&&`/`||` (hoje avalia ambos), structs
comparáveis por valor ou ponteiros.

Antes: split `forge-core`/`forge-cli` (item F8) — a lib pura permite o LSP
consumir o mesmo pipeline e mantém a CLI fina.
