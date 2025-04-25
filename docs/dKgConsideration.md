# Dynamic-KG: Design Considerations Document 🧠🔄💫

## Introduction 🌟

This document outlines the design considerations for implementing the Dynamic Knowledge Graph (Dynamic-KG) for loan risk assessment. Our goal is to create an experimental framework that captures the temporal evolution of entities and their relationships, with a focus on enabling "backdate" simulations for testing and evaluation.

## Temporal Modeling Approaches 🕰️

There are three primary approaches to implementing temporal dimensions in our knowledge graph:

### Option A: Property-Based Temporal Attributes 📊

In this approach, we add temporal attributes to each node and relationship in the graph.

```cypher
CREATE (c:Company {
    id: "company_acme",
    name: "ACME Corp",
    industry: "Technology",
    valid_from: datetime("2023-01-01"),
    valid_to: datetime("2023-12-31"),
    confidence: 0.95,
    evidence_id: "FIN-20230115-9283"
})
```

**Pros:**
- Simple implementation that works with Neo4j Community Edition
- Lower storage overhead compared to other approaches
- Straightforward schema that maintains clarity

**Cons:**
- Requires custom query patterns for temporal reasoning
- More complex queries when trying to understand historical evolution
- Historical states aren't explicitly represented as entities

### Option B: Versioned Nodes and Relationships 🧩

This approach creates new versions of nodes and relationships as they change over time, connecting versions with explicit relationships.

```cypher
// Create initial company node
CREATE (c1:Company:Version {
    id: "company_acme_v1",
    canonical_id: "company_acme",
    name: "ACME Corp",
    industry: "Technology",
    valid_from: datetime("2023-01-01"),
    valid_to: datetime("2023-06-30"),
    confidence: 0.95
})

// Create updated version after a change
CREATE (c2:Company:Version {
    id: "company_acme_v2",
    canonical_id: "company_acme",
    name: "ACME Corporation", // Name changed
    industry: "Technology",
    valid_from: datetime("2023-07-01"),
    valid_to: null,
    confidence: 0.98
})

// Connect versions
CREATE (c1)-[:NEXT_VERSION]->(c2)
CREATE (c2)-[:PREVIOUS_VERSION]->(c1)
```

**Pros:**
- Preserves complete history with explicit version nodes
- Easier to track the evolution of entities over time
- Simpler queries for retrieving historical states
- Better for auditing and explaining changes

**Cons:**
- Graph grows larger over time as entities change
- More complex relationship patterns
- Requires additional logic to determine current version

### Option C: Snapshot-Based Approach 📸

This approach stores complete graph snapshots at different time points, possibly in separate subgraphs.

```cypher
// Create snapshot container nodes
CREATE (s1:Snapshot {
    id: "snapshot_2023Q1",
    timestamp: datetime("2023-03-31"),
    description: "Q1 2023 Snapshot"
})

CREATE (s2:Snapshot {
    id: "snapshot_2023Q2",
    timestamp: datetime("2023-06-30"),
    description: "Q2 2023 Snapshot"
})

// Create company nodes in each snapshot
CREATE (c1:Company {
    id: "company_acme_2023Q1",
    name: "ACME Corp",
    revenue: 1000000,
    snapshot_id: "snapshot_2023Q1"
})

CREATE (c2:Company {
    id: "company_acme_2023Q2",
    name: "ACME Corporation",
    revenue: 1200000,
    snapshot_id: "snapshot_2023Q2"
})

// Link to snapshots
CREATE (s1)-[:CONTAINS]->(c1)
CREATE (s2)-[:CONTAINS]->(c2)
```

**Pros:**
- Easy to query the complete state at any specific time
- Simplifies time-point analysis
- Natural fit for simulation "replay" functionality
- Clear separation between time periods

**Cons:**
- Highly storage intensive with significant data duplication
- Complex to manage relationships between versions
- Challenging to query across time periods

## Recommended Hybrid Approach 🔄

For our POC/MVP, we recommend a hybrid approach combining elements from Options A and B:

1. **Base Layer (Option A)**: Use property-based temporal attributes for all entities and relationships
2. **Version Tracking (Option B)**: For key entities (companies, loan applications), maintain explicit version chains
3. **Simulation Snapshots**: Create lightweight snapshot markers at significant time points for simulation control, without duplicating the entire graph

This approach gives us:
- Reasonable storage efficiency
- Clear traceability of entity changes over time
- Strong support for backdating simulations

## Confidence Scoring Implementation 📏

Our confidence scoring system will be implemented as follows:

```cypher
CREATE (c:Company {
    id: "company_acme",
    name: "ACME Corp",
    valid_from: datetime("2023-01-01"),
    valid_to: datetime("2023-12-31"),
    confidence: 0.95,
    confidence_factors: {
        source_reliability: 0.98,
        extraction_confidence: 0.92,
        temporal_relevance: 0.97,
        corroboration_level: 0.93
    },
    evidence_id: "FIN-20230115-9283",
    evidence_type: "financial_statement",
    evidence_source: "quarterly_report_q1_2023"
})
```

Key elements:
- Core confidence score (0.0-1.0)
- Decomposed confidence factors for explainability
- Evidence attribution linking to source data
- Temporal relevance that can decay over time

## Backdating Simulation Design 🎬

The backdating simulation capability is crucial for experimentation and will include:

### 1. Timeline Construction 📅

- Create a chronological timeline of all events, data points, and knowledge graph changes
- Index significant time points where major changes or events occurred
- Tag events with their source type (market event, company event, loan event)

### 2. Simulation Controller ⏮️⏯️⏭️

```python
class TimelineController:
    def __init__(self, graph_connector, start_date=None):
        self.graph = graph_connector
        self.current_date = start_date or earliest_date_in_graph()
        self.event_timeline = load_events_in_order()
        
    def advance_to_date(self, target_date):
        """Move simulation time forward to specified date."""
        # Process all events between current_date and target_date
        
    def rewind_to_date(self, target_date):
        """Reset simulation to earlier point in time."""
        # Reset graph to state at target_date
        
    def inject_event(self, event_data, event_date):
        """Add a synthetic event to the timeline."""
        # Insert event and propagate its effects
        
    def get_graph_state(self):
        """Get current graph state at simulation time."""
        # Return subgraph representing current state
```

### 3. Counterfactual Analysis 🔮

- Fork the timeline at any point to create alternative scenarios
- Inject hypothetical events (e.g., "What if company X had filed for bankruptcy?")
- Compare outcomes between baseline and counterfactual scenarios
- Measure the impact of information timing (e.g., "What if we had known about this event 30 days earlier?")

## Experimental Framework Structure 🧪

The experimental framework will be organized as follows:

### 1. Baseline Experiments 📊

- Start with loan applications from 24 months ago
- Process with only information available at that time
- Evaluate risk assessment accuracy against known outcomes

### 2. Information Timing Experiments ⏱️

- Vary when different information becomes available
- Test "early warning" capabilities by injecting events earlier
- Measure how prediction quality changes with information timing

### 3. Signal Propagation Experiments 📡

- Track how events affecting one company propagate to others
- Measure impact propagation through different relationship types
- Identify optimal propagation paths and distance cutoffs

### 4. Confidence Evolution Experiments 📈

- Track how confidence scores evolve as new evidence arrives
- Test different confidence decay functions
- Measure impact of corroborating evidence on decision quality

## Implementation Phasing 🚀

### Phase 1: Basic Graph Schema 🏗️

1. Set up Neo4j Community Edition
2. Implement core node and relationship types
3. Create basic loader from existing dataStore
4. Build simple visualization in Streamlit

### Phase 2: Temporal Extensions ⏰

1. Add temporal properties to all entities
2. Implement version chains for key entities
3. Create timeline visualization
4. Build snapshot marker system

### Phase 3: Simulation Engine 🎮

1. Implement timeline controller
2. Create event injection mechanism
3. Build counterfactual analysis tools
4. Develop simulation visualization

### Phase 4: Experimental Framework 🔬

1. Implement baseline experiments
2. Create information timing experiments
3. Build signal propagation analysis
4. Develop confidence evolution tracking

## Conclusion 🌈

The Dynamic-KG approach provides a powerful framework for exploring how knowledge evolves over time and how that evolution affects decision-making. By focusing on the temporal dimension and building robust simulation capabilities, we can test hypotheses about information value and create more adaptive, context-aware risk assessment systems.

The hybrid approach balances implementation complexity with experimental flexibility, giving us a solid foundation to explore the core innovations of the Dynamic-KG concept while keeping the POC/MVP manageable.

---

*"The best way to predict the future is to simulate it."* 📈