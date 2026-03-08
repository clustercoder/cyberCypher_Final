"""RAG Knowledge Base — stores network operations runbooks and retrieves
relevant context for the reasoning agent.

Uses:
  - OpenAIEmbeddings (text-embedding-3-small)
  - FAISS vector store
  - RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=50)
"""
from __future__ import annotations

import os
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ---------------------------------------------------------------------------
# Runbook content
# ---------------------------------------------------------------------------

_RUNBOOKS: list[tuple[str, str]] = [
    (
        "Congestion Response Playbook",
        """
Congestion Response Playbook
=============================

Overview
--------
Link congestion is the most common cause of SLA breaches in ISP networks.
This playbook covers end-to-end response from detection through resolution.

Detection Criteria
------------------
Congestion is confirmed when any of the following thresholds are exceeded:
- Link utilization > 85% sustained for ≥ 3 consecutive minutes
- Packet loss > 1% on a core or aggregation link
- RTT latency > 2.5× the per-link baseline for ≥ 5 minutes
- Buffer drop rate > 50 drops/min on the upstream router

Immediate Response (0–5 minutes)
---------------------------------
1. Identify the congested link via topology graph.
2. Check upstream links for cascading overload (look for ≥ 70% utilization
   on adjacent links).
3. Verify traffic composition: run `show interface` and check for unexpected
   traffic surges (possible DDoS — cross-reference with DDoS Playbook).
4. Apply temporary rate limiting at the ingress router to cap the traffic
   class causing the surge. Rate-limit cap: 70% of link capacity.

Traffic Rerouting (5–15 minutes)
---------------------------------
1. Identify alternate paths using the topology causal graph.
2. Update TE weights to shift 30–50% of traffic to the least-loaded path.
3. Monitor the target link for at least 3 minutes post-reroute to confirm
   utilization drops below 80%.
4. If rerouting is not available (hub-and-spoke topology with a single path),
   proceed immediately to capacity scaling.

Rate Limiting as Temporary Measure
------------------------------------
- Rate limiting should cap ingress traffic to 70% of link capacity.
- Apply per-source-prefix rate limits to avoid penalizing all customers.
- Set a TTL on the rate limit: auto-expire after 30 minutes unless manually
  extended by a network engineer.
- Monitor customer complaints in the ticketing system during rate limiting.

Escalation Criteria
--------------------
- Escalate to Tier 2 if: congestion persists > 20 min after reroute.
- Escalate to capacity planning team if: 70% utilization is sustained for
  more than 4 hours (indicating a capacity upgrade is needed).

SLA Impact
-----------
- Enterprise customers (99.99% SLA): notify immediately if congestion
  causes latency > 10ms above baseline or packet loss > 0.1%.
- Residential customers (99.9% SLA): notify if outage exceeds 5 minutes.

Typical Resolution Time
------------------------
- Reroute available: 15–20 minutes end-to-end.
- No alternate path, rate limiting only: 20–30 minutes.
- Requires capacity upgrade: 2–4 hours (emergency provisioning).
""",
    ),
    (
        "BGP Route Flap Troubleshooting",
        """
BGP Route Flap Troubleshooting
================================

Overview
--------
BGP instability causes prefix reachability issues and can trigger traffic
black-holing. Route flaps must be diagnosed quickly to prevent cascading
failures across peering relationships.

Detection
----------
BGP instability is identified by:
- Route change frequency > 10 updates/minute on a single peer session.
- Same prefix withdrawn and re-announced ≥ 3 times in 5 minutes.
- BGP hold timer expiry on a peering session (event logged by router).
- Sudden drop in prefix count from a peer (≥ 20% reduction).

Diagnosis Steps
---------------
1. Check BGP session state: `show bgp summary` on the affected router.
2. Identify the flapping prefix and its originating AS using BGP route table.
3. Determine flap source:
   a. Local misconfiguration (check for recent config changes in audit log).
   b. Upstream provider instability (check NTT/Cogent/Telia NOC feeds).
   c. Route reflector issue (verify RR peering health if internal iBGP flap).
4. Check dampening parameters:
   - If dampening is enabled, verify the half-life and reuse thresholds are
     appropriate (recommended: half-life=15min, reuse=750, suppress=2000).
   - Excessive dampening can prevent legitimate route recovery.

Rollback Procedures
--------------------
For bad route advertisements:
1. Withdraw the misconfigured prefix immediately via `clear ip bgp` on the
   affected neighbor.
2. Restore the last-known-good route configuration from the version-
   controlled config backup (git-based config management).
3. Verify route re-convergence across all peering sessions before closing.
4. Document the root cause and fix in the incident post-mortem.

Upstream Provider Issues
-------------------------
- Contact upstream NOC with the flapping AS path and affected prefix list.
- Request BGP graceful restart on the provider side if possible.
- Consider temporarily de-preferring routes from the unstable peer by
  lowering the local-preference value.

Internal Misconfiguration
--------------------------
- Common causes: incorrect redistribute statements, absent route filters,
  wrong next-hop for iBGP routes.
- Verify with diff against last committed config in the change management
  system.
""",
    ),
    (
        "DDoS Mitigation Procedures",
        """
DDoS Mitigation Procedures
============================

Overview
--------
Distributed Denial of Service attacks are identified by sudden, sustained
traffic surges that overwhelm link capacity and degrade service for all
customers. Rapid detection and mitigation are critical.

Detection Criteria
------------------
A DDoS attack is suspected when:
- Traffic volume increases > 2× baseline within 5 minutes.
- Packet-per-second rate spikes without a corresponding increase in unique
  source IPs (amplification attack indicator).
- ICMP or UDP flood pattern detected by flow analytics.
- A specific customer or IP range reports complete service loss.
- Edge router CPU exceeds 90% due to stateful connection tracking overload.

Immediate Response (0–3 minutes)
----------------------------------
1. Confirm it is DDoS (not organic traffic growth): check source IP diversity,
   packet type distribution, and targeted destination IPs.
2. Identify the attack vector: volumetric (UDP/ICMP flood), protocol
   (SYN flood), or application layer (HTTP flood).
3. Apply rate limiting at the peering point: cap ingress from affected
   prefixes to 30% of normal baseline.

Traffic Scrubbing (3–10 minutes)
----------------------------------
1. Redirect attack traffic to the scrubbing center via BGP blackhole
   community or FlowSpec rule.
2. Scrubbing appliance drops attack traffic; legitimate traffic is
   re-injected into the network.
3. Monitor scrubbing effectiveness: target > 95% attack traffic dropped,
   < 2% legitimate traffic dropped.

Escalation to Upstream Provider (10+ minutes)
----------------------------------------------
1. If scrubbing center is overwhelmed (> 80% scrubbing capacity), escalate
   to upstream provider DDoS mitigation service.
2. Provide upstream provider with: attack vector, source ASN, target prefix,
   attack volume in Gbps and Mpps.
3. Request upstream blackhole or rate limiting to drop attack traffic closer
   to its source.

Blackhole Routing (Last Resort)
---------------------------------
- Only apply blackhole routing if the attack is causing collateral damage to
  other customers on the same infrastructure.
- Notify the affected customer before applying blackhole (service will be
  completely unavailable during blackhole).
- Document the blackhole duration and customer notification time.

Customer Communication
-----------------------
- Notify affected enterprise customers within 5 minutes of confirmed DDoS.
- Template: "We are currently experiencing a DDoS attack targeting your
  network segment. Our mitigation procedures are active and we expect
  service restoration within [X] minutes."
- Send status updates every 15 minutes until resolved.
""",
    ),
    (
        "Hardware Failure Escalation Matrix",
        """
Hardware Failure Escalation Matrix
====================================

Overview
--------
Hardware failures on network devices require a structured escalation process
to minimize customer impact. This matrix defines thresholds, response
procedures, and customer impact by device type.

Temperature Thresholds
-----------------------
- Warning: 65°C — log alert, schedule inspection during next maintenance window.
- Critical: 75°C — trigger automatic fan speed increase, alert on-call engineer.
- Emergency: 82°C — initiate graceful traffic failover, schedule emergency
  hardware replacement. Risk of permanent hardware damage above this threshold.

CPU Thresholds
--------------
- Warning: 85% for > 5 minutes — check for routing protocol storms or high
  BGP update rates.
- Critical: 95% for > 2 minutes — risk of control plane failure (BGP sessions
  dropping, OSPF adjacency loss). Immediately reduce routing protocol load.
- Action: Redistribute traffic to adjacent routers; prepare for planned
  failover to backup hardware.

Memory Thresholds
-----------------
- Warning: 85% — check for memory leaks in recent software version; plan
  reload during maintenance window.
- Critical: 95% — risk of process crashes. Emergency maintenance required.

Buffer Drop Thresholds
-----------------------
- Warning: > 100 drops/minute — indicates sustained congestion; review QoS
  policies and traffic engineering.
- Critical: > 500 drops/minute — customer-impacting; trigger congestion
  response playbook in parallel.

Maintenance vs Emergency Response
-----------------------------------
Schedule maintenance window when:
  - Issue is non-urgent (warning thresholds only)
  - Traffic can be drained in a controlled manner
  - Spare equipment is on-site
Declare emergency when:
  - Critical threshold exceeded on a core or aggregation node
  - No redundant path exists for affected customer traffic
  - Temperature > 80°C

Spare Equipment Procedures
---------------------------
1. Verify spare equipment availability in site inventory.
2. If no spare on-site: contact hardware vendor for emergency 4-hour delivery.
3. Pre-configure the replacement device from the backup configuration (stored
   in the network configuration management system).
4. Hot-swap where supported; cold-swap requires 15–45 minute outage.

Customer Impact by Device Type
--------------------------------
- Core router failure: affects ALL customers in the region.
- Aggregation router failure: affects all customers connected via that agg node
  (typically 10,000–50,000 customers).
- Edge router failure: affects customers connected to that specific edge node
  (typically 5,000–20,000 customers).
- Peering router failure: affects BGP reachability to specific upstream ASNs;
  may not directly impact customers if alternate peering paths exist.
""",
    ),
    (
        "SLA Breach Response Protocol",
        """
SLA Breach Response Protocol
==============================

Overview
--------
Service Level Agreements define the minimum availability and performance
guarantees to customers. A breach occurs when actual service delivery falls
below the contracted threshold.

SLA Tiers
----------
- Enterprise (Tier 1): 99.99% availability = ≤ 52.6 minutes downtime/year.
  Includes latency SLA: < 10ms RTT within the network core.
- Business (Tier 2): 99.9% availability = ≤ 8.76 hours downtime/year.
- Residential (Tier 3): 99.9% availability, no latency guarantee.

Breach Detection
-----------------
A breach is triggered when:
- Cumulative downtime in the current month exceeds the SLA allowance.
- Latency exceeds the contracted threshold for > 15 consecutive minutes
  (Enterprise Tier 1 only).
- Packet loss > 1% sustained for > 30 minutes for Tier 1 customers.

Required Notification Timelines
---------------------------------
- Tier 1 Enterprise: notify within 15 minutes of confirmed breach.
- Tier 2 Business: notify within 1 hour of confirmed breach.
- Tier 3 Residential: notify via status page within 2 hours of confirmed breach.
- Regulatory notification (if breach affects > 10,000 customers): within 24
  hours per regulatory framework.

Penalty Calculations
---------------------
- Tier 1: 10% monthly fee credit per hour of downtime exceeding SLA.
  Maximum credit: 100% of monthly fee.
- Tier 2: 5% monthly fee credit per 4 hours of downtime exceeding SLA.
- Tier 3: Credits applied automatically for outages > 4 hours (Tier 3 credit
  policy per residential terms of service).

Root Cause Report Requirements
--------------------------------
For Tier 1 Enterprise customers:
- Preliminary report within 24 hours: incident timeline, initial root cause.
- Final report within 5 business days: full RCA, contributing factors, fix
  deployed, preventive measures.
- Report must be signed off by the Network Engineering Manager.

Credit Issuance Procedures
----------------------------
1. Calculate downtime from monitoring system (confirmed by customer ticket).
2. Compute credit amount per the penalty matrix above.
3. Apply credit to next billing cycle (automated via billing system).
4. Notify the account manager to communicate credit to the customer.
5. Document in the SLA tracking system.
""",
    ),
    (
        "Traffic Engineering Best Practices",
        """
Traffic Engineering Best Practices
=====================================

Overview
--------
Traffic Engineering (TE) ensures optimal utilization of network capacity
by controlling how traffic flows through the network. Effective TE prevents
hotspots, maximizes throughput, and minimizes latency.

TE Weight Optimization
-----------------------
- TE weights control OSPF/ISIS path preference. Lower weight = preferred path.
- Adjust TE weights to achieve ≤ 60% utilization on all core links during
  peak hours (target: 50% average across core).
- Do not set any single link weight to force 100% traffic concentration
  (anti-hotspot rule).
- Weight changes must be staged: change one link at a time and monitor for
  3 minutes before proceeding.

ECMP Considerations
--------------------
- Equal-Cost Multi-Path routing distributes load across equal-weight paths.
- Ensure an even number of parallel paths (2 or 4) to maximize ECMP benefit.
- Verify flow-based hashing is enabled (not per-packet) to avoid out-of-order
  packet delivery.
- For flows sensitive to jitter (VoIP, real-time video), pin to a single path
  using traffic class-based ECMP.

Time-of-Day Traffic Patterns
------------------------------
- Peak hours: 18:00–23:00 local time (residential traffic dominant).
- Off-peak: 02:00–08:00 local time (enterprise batch traffic windows).
- Adjust TE weights at peak onset (17:45) and off-peak onset (01:45) via
  scheduled config push.
- Monitor for deviations from the diurnal pattern — sudden surges outside
  peak hours indicate anomalous events (DDoS, large backup job, BGP leak).

Capacity Planning Thresholds
------------------------------
- Plan capacity upgrades when sustained utilization > 70% during peak hours
  for 3 consecutive days.
- Emergency capacity trigger: sustained > 85% for > 2 hours.
- Capacity lead time: 4–8 weeks for new fiber; 1–2 weeks for wavelength
  upgrade on existing fiber.
- Target headroom after upgrade: ≤ 50% utilization at current peak demand.

Cross-Regional Traffic Balancing
----------------------------------
- Use BGP local-preference to balance traffic across regional entry points.
- Avoid asymmetric routing (ingress and egress on different links for the
  same flow) to simplify troubleshooting.
- Monitor inter-region link utilization ratios; alert if one region carries
  > 2× the traffic of a peer region with similar customer counts.
""",
    ),
    (
        "Incident Post-Mortem Template",
        """
Incident Post-Mortem Template
===============================

Overview
--------
Post-mortems are blameless analyses of service incidents. They document what
happened, why it happened, and how to prevent recurrence. All Tier 1 and
major Tier 2 incidents require a post-mortem within 5 business days.

Timeline Reconstruction
------------------------
Record timestamps in UTC. Include:
- T+0: First alert or customer report received.
- T+N: Each significant action taken (detection, escalation, mitigation,
  resolution).
- T+Resolution: Service fully restored.
- Use monitoring system logs and the agent audit trail as primary sources.

Root Cause Analysis Format
---------------------------
1. Proximate cause: The immediate technical cause of the failure (e.g.,
   "Link CR1-CR2 utilization reached 96% causing packet loss").
2. Contributing factors: Conditions that enabled the proximate cause (e.g.,
   "No alternate path available due to planned maintenance on CR1-AGG2").
3. Triggering event: The specific event that initiated the failure chain (e.g.,
   "BGP route advertisement from PEER1 shifted 40% of transit traffic to CR1-CR2").
4. Detection gap: Was the incident detected by automated monitoring or by
   customers? If customer-reported, document why monitoring missed it.

Contributing Factors
---------------------
Analyze across these dimensions:
- People: Were procedures followed? Was training adequate?
- Process: Were runbooks up-to-date? Were escalation paths clear?
- Technology: Were thresholds correctly configured? Were there tool gaps?
- External: Upstream provider issue? Unexpected traffic growth?

Action Items
-------------
Each action item must have:
- Owner (team or individual)
- Due date
- Success criterion (how will we know it's done?)
- Priority: P1 (within 1 week), P2 (within 1 month), P3 (within 1 quarter)

Prevention Measures
--------------------
Document preventive controls added after the incident:
- New monitoring thresholds or alerts.
- Runbook updates.
- Capacity upgrades planned.
- Architectural changes (e.g., adding redundant paths).
- Automated responses added to the AI agent playbook.

Key Metrics to Track
---------------------
- MTTD (Mean Time to Detect): From first anomaly to first alert.
  Target: < 60 seconds for automated detection.
- MTTM (Mean Time to Mitigate): From first alert to mitigation applied.
  Target: < 5 minutes for automated response, < 15 minutes for human-assisted.
- Blast Radius: Number of customers affected.
  Target: < 1% of total customer base for any single incident.
- Customer Impact Score: Severity × Duration × Customers affected.
- Recurrence Rate: Same root cause appearing within 90 days (target: 0).
""",
    ),
]


# ---------------------------------------------------------------------------
# RAGKnowledgeBase
# ---------------------------------------------------------------------------

class RAGKnowledgeBase:
    """FAISS-backed vector store for network operations runbooks.

    Requires ``OPENAI_API_KEY`` in the environment.
    """

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Set it in .env or the environment."
            )

        self._embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        self._store: FAISS | None = None
        self._sources: list[str] = []

        self._populate_default_runbooks()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_document(self, title: str, content: str) -> None:
        """Chunk and embed a document, then add it to the FAISS store."""
        chunks = self._splitter.split_text(content)
        metadatas = [{"source": title} for _ in chunks]

        if self._store is None:
            self._store = FAISS.from_texts(
                chunks, self._embeddings, metadatas=metadatas
            )
        else:
            self._store.add_texts(chunks, metadatas=metadatas)

        if title not in self._sources:
            self._sources.append(title)

    def query(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Retrieve the top_k most relevant chunks for a question.

        Returns
        -------
        list of dicts with keys: ``content``, ``source``, ``relevance_score``
        """
        if self._store is None:
            return []

        results = self._store.similarity_search_with_relevance_scores(
            question, k=top_k
        )
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "relevance_score": round(float(score), 4),
            }
            for doc, score in results
        ]

    def add_incident_learning(self, incident_summary: str) -> None:
        """Add a resolved incident summary as a new learning document."""
        title = f"Incident Learning: {incident_summary[:60]}..."
        self.add_document(title, incident_summary)

    def get_document_count(self) -> int:
        """Return the total number of chunks in the vector store."""
        if self._store is None:
            return 0
        return self._store.index.ntotal

    def get_all_sources(self) -> list[str]:
        """Return all document titles loaded into the knowledge base."""
        return list(self._sources)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _populate_default_runbooks(self) -> None:
        for title, content in _RUNBOOKS:
            self.add_document(title, content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("\n── RAG Knowledge Base ────────────────────────────────────────────\n")
    print("  Initializing knowledge base (embedding runbooks)…")
    kb = RAGKnowledgeBase()
    n = kb.get_document_count()
    print(f"  ✅ RAG Knowledge Base ready with {n} chunks from {len(kb.get_all_sources())} documents")
    print(f"  Sources: {kb.get_all_sources()}\n")

    queries = [
        "How should I handle high utilization on a backbone link?",
        "What are the DDoS mitigation steps?",
    ]

    for q in queries:
        print(f"  Query: \"{q}\"")
        results = kb.query(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] source={r['source']}  score={r['relevance_score']:.4f}")
            print(f"      {r['content'][:200].strip()}…")
        print()
