# Project History

The story behind WakeBuilder's development.

---

## Origins

WakeBuilder was born from a simple frustration: creating custom wake word detection models was unnecessarily complicated, expensive, and required sending voice data to the cloud.

### The Problem

Before WakeBuilder, developers and hobbyists who wanted custom wake words faced:

- **Commercial solutions** requiring subscriptions and cloud dependency
- **Complex ML pipelines** requiring deep expertise
- **Large datasets** needing thousands of voice recordings
- **Privacy concerns** about sending voice data to external services

### The Vision

What if anyone could create a custom wake word model:

- 🏠 **Locally** - on their own machine
- 🔒 **Privately** - without sharing voice data
- 🎯 **Simply** - without ML expertise
- ♿ **Accessibly** - for users with unique speech patterns

---

## Development Timeline

### Phase 1: Foundation (Research & Architecture)

**Focus**: Selecting the right technology stack

- Evaluated multiple base speech models
- Chose Audio Spectrogram Transformer (AST) for transfer learning
- Designed the three-layer architecture
- Set up audio preprocessing with librosa

**Key Decision**: Using transfer learning with AST enabled few-shot learning—training effective models from just a few recordings.

### Phase 2: Training Pipeline

**Focus**: Data augmentation and model training

- Integrated Piper TTS for synthetic voice generation
- Developed the hard negative generation algorithm
- Implemented the AST-based classifier architecture
- Created the training loop with early stopping

**Key Innovation**: The hard negative generator that creates phonetically similar words proved critical for reducing false positives.

### Phase 3: Backend API

**Focus**: FastAPI-based service layer

- Built RESTful API endpoints for training management
- Implemented WebSocket for real-time testing
- Created the job management system
- Developed model storage and retrieval

### Phase 4: Web Interface

**Focus**: User-friendly frontend

- Designed the training wizard
- Implemented audio recording with Web Audio API
- Created real-time training progress visualization
- Built the model testing interface

### Phase 5: Polish & Optimization

**Focus**: Performance and user experience

- Added multiple TTS providers (Edge, Kokoro, Coqui)
- Implemented negative data caching for faster training
- Optimized memory usage
- Added ONNX export for deployment

---

## Key Decisions

### Why AST?

The Audio Spectrogram Transformer was chosen because:

1. **Pre-trained on Speech Commands**: Already understands speech patterns
2. **Transfer learning friendly**: Small classifier head is enough
3. **Open source**: MIT license, no restrictions
4. **Well-supported**: Hugging Face transformers integration

### Why Multiple TTS Engines?

Different TTS engines provide different strengths:

| Engine | Strength |
|--------|----------|
| **Piper** | Fast, offline, high quality |
| **Edge** | Largest voice variety |
| **Kokoro** | Natural prosody |
| **Coqui** | Multi-speaker models |

### Why Local-First?

Privacy was a core requirement:

- Voice data is highly personal
- Cloud dependency creates vendor lock-in
- Offline operation is more reliable
- Users own their trained models

---

## Technical Evolution

### Model Architecture

**Initial**: Simple MLP classifier on AST embeddings

**Current**: Enhanced classifier with optional:

- Self-attention pooling
- Squeeze-and-excitation blocks
- Temporal convolutional networks

### Loss Function

**Initial**: Binary cross-entropy

**Current**: Focal loss with label smoothing

- Better handling of class imbalance
- Focus on hard examples
- Improved calibration

### Data Augmentation

**Initial**: Basic speed and volume changes

**Current**: Comprehensive augmentation:

- TTS synthesis (440+ voices)
- Speed variations (0.8x-1.5x)
- Pitch shifting (±2 semitones)
- Background noise injection
- Volume normalization

---

## Lessons Learned

### What Worked Well

1. **Transfer learning**: Dramatically reduced data requirements
2. **Hard negatives**: Critical for preventing false positives
3. **Multi-TTS**: Diverse voices improved generalization
4. **Web interface**: Made the technology accessible

### Challenges Overcome

1. **Memory management**: TTS generation is memory-intensive
   - Solution: Lazy loading and cleanup after synthesis

2. **Training stability**: Early models overfit quickly
   - Solution: Aggressive dropout, mixup, focal loss

3. **False positives**: Similar words triggered detection
   - Solution: Hard negative generation algorithm

---

## Current State

WakeBuilder is currently in **Beta** status:

| Component | Status |
|-----------|--------|
| Training pipeline | ✅ Stable |
| Web interface | ✅ Stable |
| AST integration | ✅ Stable |
| Docker deployment | 🚧 In progress |
| Default models | 🚧 In progress |

---

## Contributors

WakeBuilder was developed and is maintained by:

**Sami RAJICHI** - Creator and sole developer

*While contributions are welcome, please understand that response times for reviewing PRs and issues may vary as this is a solo project.*

---

## Future Direction

See the [Roadmap](roadmap.md) for planned features and improvements.

---

## Acknowledgments

WakeBuilder builds upon outstanding open-source work:

- **[MIT AST](https://github.com/YuanGongND/ast)** - Audio Spectrogram Transformer
- **[Piper TTS](https://github.com/rhasspy/piper)** - Local neural TTS
- **[Edge TTS](https://github.com/rany2/edge-tts)** - Microsoft neural voices
- **[Kokoro TTS](https://github.com/hexgrad/kokoro)** - High-quality English TTS
- **[Coqui TTS](https://github.com/coqui-ai/TTS)** - Multi-speaker TTS
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[FastAPI](https://fastapi.tiangolo.com/)** - Web framework
- **[librosa](https://librosa.org/)** - Audio processing

See [Acknowledgments](acknowledgments.md) for the full list.
