# Network Troubleshooting Knowledge Base for AI Agent

Source: Switching Design Best Practices and Case Studies -- Dr. Peter
Welcher

## Overview

This knowledge base is designed for an AI agent that monitors enterprise
networks and automatically detects, diagnoses, and mitigates switching
and routing issues.

The system focuses on: - Campus network architecture - Layer2 / Layer3
switching behavior - Spanning Tree monitoring - VLAN management -
EtherChannel issues - Broadcast storm detection - Network failure case
studies

------------------------------------------------------------------------

# 1. Campus Network Architecture

## Three Tier Model

### Access Layer

-   IDF switches
-   End user connectivity
-   VLAN access ports

### Distribution Layer

-   MDF switches
-   Policy enforcement
-   Inter‑VLAN routing

### Core Layer

-   High speed switching backbone
-   WAN/Internet connectivity

Network flow:

Users → Access Switch → Distribution Switch → Core Switch → WAN

Benefits: - Scalability - Fault isolation - Simplified troubleshooting

------------------------------------------------------------------------

# 2. L2/L3 Hierarchical Design

Classic campus networks use:

-   Layer2 VLAN switching at access
-   Layer3 routing at distribution
-   Spanning Tree Protocol for loop prevention
-   HSRP for gateway redundancy

Best practices:

-   Odd/even VLAN STP load balancing
-   Controlled VLAN trunk propagation
-   VLAN pruning on trunks

Detection Rules:

  Symptom            Possible Cause
  ------------------ --------------------------
  STP instability    Root bridge misplacement
  VLAN flooding      Too many VLANs on trunks
  Slow convergence   Poor STP topology

------------------------------------------------------------------------

# 3. Modern L3 Campus Design

Modern networks push Layer3 closer to the edge.

Changes include: - Routing on server switches - Routed point‑to‑point
links - Smaller Layer2 domains

Benefits: - Reduced broadcast domains - Faster convergence - Better
failure isolation

Agent rule:

IF large L2 domain detected\
THEN recommend L3 segmentation

------------------------------------------------------------------------

# 4. Spanning Tree Risk Management

Common STP problems:

-   Unknown root bridge
-   User switches creating loops
-   Broadcast storms

Indicators:

-   High broadcast traffic
-   Frequent topology changes
-   Switch CPU spikes

Monitoring signals:

-   MAC address table instability
-   STP recalculation events

------------------------------------------------------------------------

# 5. VLAN Management

Best practices:

-   Create VLAN before assigning ports
-   Move ports before deleting VLAN
-   Avoid VLAN surprises

Failure symptom:

Ports become **errdisabled** when VLANs are deleted improperly.

------------------------------------------------------------------------

# 6. VTP Best Practices

Recommended configuration:

Two switches → VTP Servers\
All others → VTP Clients

OR

All switches → Transparent Mode

Purpose:

Prevent catastrophic VLAN deletion caused by high VTP revision numbers.

Agent rule:

IF unexpected VLAN deletion detected\
CHECK VTP revision numbers

------------------------------------------------------------------------

# 7. EtherChannel Monitoring

EtherChannel provides:

-   Higher bandwidth
-   Redundancy
-   Load balancing

Typical aggregated bandwidth:

1 Gbps\
2 Gbps\
4 Gbps\
10 Gbps

Common configuration errors:

-   Mismatched channel groups
-   Partial link aggregation
-   Asymmetric configurations

Agent rule:

IF EtherChannel members mismatch\
THEN flag configuration inconsistency

------------------------------------------------------------------------

# 8. Broadcast Storm Detection

Flat networks create excessive broadcast traffic.

Example scenario:

-   Single VLAN
-   500+ users
-   2 Mbps broadcast traffic

Symptoms:

-   Network slowdown
-   Router CPU spikes
-   Packet drops

Mitigation:

-   VLAN segmentation
-   Layer3 routing

------------------------------------------------------------------------

# 9. Root Bridge Misconfiguration

Problem:

Access switches sometimes become STP root bridge because they have the
lowest MAC address.

Solution:

Configure root bridge explicitly.

Example:

spanning-tree vlan `<id>`{=html} priority `<value>`{=html}

Agent rule:

IF root bridge located on access switch\
THEN raise alert

------------------------------------------------------------------------

# 10. User‑Introduced Network Loops

Users may connect unmanaged switches.

Effects:

-   STP instability
-   Broadcast storms
-   Network outages

Prevention features:

-   BPDU Guard
-   Root Guard
-   PortFast

Agent indicators:

-   MAC flapping
-   Sudden STP topology changes

------------------------------------------------------------------------

# 11. Data Center Loop Incident

Observed scenario:

Adding new switches caused a spanning tree loop resulting in **700
servers going offline**.

Root cause:

EtherChannel configuration mismatch.

Prevention:

-   Validate configurations before deployment
-   Use change control
-   Test in lab environment

------------------------------------------------------------------------

# 12. Router-on-a-Stick Bottleneck

Symptoms:

-   Slow WAN
-   High router CPU
-   Packet drops

Cause:

Traffic routed inefficiently through WAN router instead of staying
within VLAN.

Fix:

-   Correct DNS resolution
-   Contain traffic inside VLAN
-   Upgrade to Layer3 switching

------------------------------------------------------------------------

# 13. CAM and ARP Timer Mismatch

Issue:

CAM table entries expire before ARP entries.

Result:

Unknown unicast flooding.

Solution:

Set CAM timer equal to ARP timer.

Agent rule:

IF unknown unicast flooding detected\
CHECK CAM vs ARP timers

------------------------------------------------------------------------

# 14. Multicast Network Failures

Symptoms:

-   MDF switch rebooting
-   Large multicast routing tables

Cause:

Uncontrolled multicast sources.

Mitigation:

-   PIM Sparse Mode
-   Auto RP
-   Multicast scoping

------------------------------------------------------------------------

# 15. High Availability

Requirements:

-   Redundant supervisors
-   Same firmware version
-   Failover testing

Common issue:

Supervisors running different software versions.

Agent rule:

Detect firmware mismatch and trigger alert.

------------------------------------------------------------------------

# 16. Layer2 Security Best Practices

Important controls:

-   BPDU Guard
-   Root Guard
-   UDLD
-   VLAN segmentation

Critical rules:

-   Do NOT use VLAN 1
-   Separate management VLAN
-   Configure native VLAN on trunks

------------------------------------------------------------------------

# 17. AI Monitoring Rules

## STP Monitoring

Triggers:

-   Root bridge change
-   Frequent topology change
-   MAC flapping

Response:

Identify loop source and isolate port.

------------------------------------------------------------------------

## Broadcast Monitoring

Triggers:

-   Broadcast traffic above threshold
-   Switch CPU spike

Response:

Locate VLAN source and isolate device.

------------------------------------------------------------------------

## EtherChannel Monitoring

Triggers:

-   Channel mismatch
-   Link imbalance

Response:

Disable misconfigured link and rebalance traffic.

------------------------------------------------------------------------

# 18. Auto Remediation Playbooks

## STP Loop

1.  Detect loop\
2.  Identify source port\
3.  Disable offending interface\
4.  Notify administrator\
5.  Enable BPDU Guard

------------------------------------------------------------------------

## Broadcast Storm

1.  Detect affected VLAN\
2.  Locate broadcast source\
3.  Isolate device\
4.  Recommend VLAN segmentation

------------------------------------------------------------------------

## VLAN Misconfiguration

1.  Detect errdisabled ports\
2.  Verify VLAN existence\
3.  Restore VLAN or reassign ports

------------------------------------------------------------------------

# 19. Network Design Principles

Reliable enterprise networks should:

-   Minimize Layer2 domains
-   Use Layer3 routing where possible
-   Avoid flat networks
-   Control STP root bridge
-   Implement redundancy
-   Maintain configuration consistency

------------------------------------------------------------------------

# 20. Key Operational Insights

Most enterprise network outages result from:

-   Human configuration errors
-   Spanning tree loops
-   EtherChannel misconfiguration
-   Broadcast storms

AI‑driven monitoring should prioritize:

-   topology awareness
-   anomaly detection
-   configuration validation
-   automated remediation
