# PTQM: Perceptual Talking Quality Model


## Overview

PTQM is an objective model for predicting the speaking-quality dimensions defined in ITU-T Recommendation P.804:

- Impact of one's own voice on speaking (IOS)
- Degradation of one's own voice (DOS)

PTQM preserves the psychoacoustic front-end of PESQM and replaces its fixed cognitive aggregation stage with a learned model operating directly on the perceptual disturbance map.


## Quick Start

PTQM requires two synchronized audio signals:

- `mic.wav`: talker's microphone signal
- `headset.wav`: signal returned to the talker's headset

The input should correspond to a single-talk interval.

python predict.py \
    --mic examples/mic.wav \
    --headset examples/headset.wav \
    --checkpoint checkpoints/ptqm.pt

## Pretrained Model

The pretrained PTQM model used in the paper is provided in:

checkpoints/ptqm.pt

