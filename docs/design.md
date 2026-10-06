# Decisões de engenharia

## Escopo
Jogo abstrato entre agentes de ataque e defesa. O cenário é um benchmark sintético independente.

## Entregas atuais
Jogo abstrato zero-sum de alocação de ataque/defesa; duas políticas Actor-Critic; coleta on-policy simultânea; PPO clipped com value loss e entropia; confronto com checkpoints e política uniforme.

## Evolução
MDP multiestado com histórico de incidentes sintético; population-based self-play; GAE; avaliação contra oponentes congelados diversos; treinamento distribuído; não executa ataques reais.

## Qualidade e colaboração
CI executa testes de comportamento. Revisões devem conferir contratos de entrada,
invariantes, divisão de dados e métricas. Evoluções de infraestrutura exigem avaliação
de custo, autorização e observabilidade. Não há recursos cloud provisionados.

## Limitações
Validar a hipótese com dados públicos licenciados ou dados autorizados antes de uso
operacional. Relatar baseline, várias sementes, intervalos de incerteza, consumo
de recursos e resultados negativos. O backlog não constitui funcionalidade implementada.
