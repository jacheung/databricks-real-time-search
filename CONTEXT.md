# Real-time search

Query autocomplete (QAC) and search results (SERP) for a retail catalog, each retrieved by L1 and reranked by a
session-aware L2 model, served from features that stay fresh to the second.

## Features

**Feature catalog**:
The one definition of every clickstream feature: name, group, aggregation, filter, window, lookup keys and online
table. Lives in `rts_shared.features`; owned by data.
_Avoid_: feature specs, feature keys, feature contract

**Feature group**:
Which entity a feature describes: `session` (in-session intent, guests too), `user` (long-term, logged in only) or
`item` (product popularity). Each group is materialized as its own Feature Views.

**Feature kind**:
Which of a group's online tables holds a feature: `window` (windowed aggregates), `category` (also keyed on
category) or `last` (latest values).
_Avoid_: table type

**Engaged event**:
A click or an add-to-cart. Searches and backs are events but not engagement.
_Avoid_: interaction
