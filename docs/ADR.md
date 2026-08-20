# ADR Skill - Architectural Decision Record


|Title|Status |Context |Decision |Consequences |Alternatives <br> Considered 
| -- | -- | -- | -- | -- | -- |
| Short description of the decision. | Tracks decision lifecycle (Proposed\|Accepted\|<br>Deprecated\|Replaced)| Explain technical issue fix/business solution achieved through the decision.|The actual architectural choice e.g. adapt MongoDB for patient visit records|Describe impact of decision: Benefits\|Risks\|Operational implications\|Long-ter trade-offs 

## Example ADR: Choosing microservices over a monolith (Simplified)

The company expects significant growth in traffic and engineering team size over the next two years.
## Title 
Adopt Microservices Architecture for the Order Platform
## Status
Accepted

## The existing monolithic architecture creates...
1. Slow deployment cycles
2. Scaling bottlenecks
3. Tight coupling between teams 
4. Increased release risk
- The business also requires faster feature delivery across multiple product domains.

## Decision

1. Split the platform into independently deployable microservices:

|Microservices||||
| -- | -- | -- | -- |
| Order service | Payment service | Inventory service | Notification service |

2. Services will communicate using REST APIs and asynchronous messaging.

## Consequences / Benefits
1. Independent service scaling
2. Faster deployments
3. Better fault isolation
4. Team autonomy

## Trade-offs
1. Increased infrastructure complexity
2. Distributed system challenges
3. More operational overhead
4. Additional monitoring requirements