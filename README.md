# Defesa cibernética em self-play

Jogo abstrato entre agentes de ataque e defesa.

**Status: MVP de pesquisa, v0.1.0.** Estudo de caso independente inspirado
em desafios dos setores relacionados à trajetória do autor (SenhaSegura · Protege).
Não representa entrega contratada, resultado real ou endosso dessas organizações.
Todos os exemplos desta versão são sintéticos e reproduzíveis.

## Executar

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m cyberplay.demo --updates 30 --seed 7
python -m unittest discover -s tests -v
```

## Implementado

Jogo abstrato zero-sum de alocação de ataque/defesa; duas políticas Actor-Critic; coleta on-policy simultânea; PPO clipped com value loss e entropia; confronto com checkpoints e política uniforme.

## Próximas etapas

MDP multiestado com histórico de incidentes sintético; population-based self-play; GAE; avaliação contra oponentes congelados diversos; treinamento distribuído; não executa ataques reais.

## Reprodutibilidade e avaliação

A demonstração grava `artifacts/report.json`, com parâmetros e métricas.
Resultados de dados sintéticos validam o fluxo; não demonstram desempenho em produção.
Use a mesma semente e orçamento computacional para comparar abordagens.
Consulte `docs/design.md` para decisões e limitações.

## Contribuição

Abra uma issue com objetivo e critério de aceite. Entregue mudanças pequenas,
revisadas e acompanhadas de testes de comportamento. Não inclua dados de clientes,
credenciais ou resultados sem proveniência. Licença MIT para código e dados sintéticos próprios.
