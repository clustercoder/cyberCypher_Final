# src/ui/dashboard/src/components/magicui/

This folder contains presentation primitives inspired by magic-UI-style patterns.

## Components

- `MagicCard.jsx`
- `ShimmerButton.jsx`
- `BorderBeam.jsx`
- `AnimatedGradientText.jsx`
- `DotPattern.jsx`

## Current Design Direction

The project currently uses a subdued, low-glare look.
Some effects are intentionally toned down or disabled in CSS for stability and readability.

## Why Keep These Components

Even with reduced effects, keeping these primitives gives flexibility:
- can re-enable specific effects for demos
- keeps styling behavior centralized
- avoids repeating visual utility code in each panel

## Practical Rule

Use these components to keep visual consistency, but prefer subtle defaults so data remains the focus.
