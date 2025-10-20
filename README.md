# Object Agents

This repository contains the implementation of **Object Agents**, a system that embodies intelligence in everyday objects to proactively assist users while maintaining their familiar affordances and identities. Rather than introducing new robotic devices, Object Agents augments familiar objects (staplers, mugs, trivets, etc.) with simple wheeled platforms, transforming them into context-aware assistants.

## Overview

Object Agents represents a step towards **unobtrusive physical AI** - systems that understand user context, predict intentions, and generate appropriate assistive behaviors for everyday objects in the physical world. The system demonstrates how familiar objects can become responsive, helpful agents that collaborate with users to enhance their daily activities.

### Core Concept

Instead of deploying dedicated robots, Object Agents transforms existing everyday objects into intelligent assistants with a perceive-reason-act loop:

1. **Perceive** - Continuously observing user activities 
2. **Reason** - Inferring user goals and potential future states from observed behaviors and maintained memory
3. **Act** - Generating contextually appropriate actions for augmented objects to assist users proactively

## System Architecture

### Core Components

- **Agent Loop** (`agent_loop.py`) - Main orchestration of the perceive-reason-act cycle
- **Perception** (`perceive.py`) - Frame analysis maintains working memory of activity narration 
- **Reasoning** (`reason.py`) - Goal generation based on observed activities and the last goal
- **Action** (`act.py`) - Plan action and check for alignment between action and goal
- **Object Detection** (`tablespace.py`) - YOLO-based object tracking and scene understanding
- **Physical Control** (`physical_object.py`) - Bluetooth control of robotic objects
- **Path Planning** (`routefinding.py`) - A* pathfinding with motion primitives
- **Display** (`display.py`) - Visualization interface 


### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd objectagentsdemo
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

4. Configure the scene 

TODO

### Object Knowledge (`prompts/object_knowledge.yaml`)

Define object properties and capabilities:

```yaml
coffee_mug:
  functionality:
    - can hold liquids for drinking
    - can serve as a container for small items
  affordances:
    - handle for grasping
    - opening at top for pouring
  # ... etc
```

## Usage

### Basic Operation

1. Start the system:
```bash
python src/main.py
```

2. The system will:
   - Initialize cameras and object detection
   - Begin observing human activities
   - Display real-time analysis in multiple windows
   - Autonomously move objects when appropriate

### Display Interface

The system provides multiple visualization windows:

- **Main Display**: Shows camera feeds and AI reasoning (Perceive/Reason/Act)
- **Developer Window**: Top-down view with object tracking and manual controls, for debugging purposes


## Hardware Setup

### Robotic Objects

TODO

The system supports Bluetooth-controlled robotic objects. Each active object needs:

- Bluetooth Low Energy (BLE) capability
- Motor control via custom characteristic UUID
- Differential drive system for movement

### Camera Setup

- **Top-down camera**: Provides planning view of the tabletop
- **Environment camera**: Capture user activities in scene

### Table Configuration

1. Set up cameras with clear view of workspace
2. Capture background image without objects
3. Create mask defining valid movement area
4. Calibrate perspective transformation coordinates
5. Train or configure YOLO models for your objects

## AI Models and Prompts

The system uses several specialized AI agents, each with carefully crafted prompts:

- **Frame Descriptor**: Analyzes individual video frames for human activities
- **Narrator**: Maintains coherent activity narrative over time
- **Goal Generator**: Infers user intentions from activity patterns  
- **Action Generator**: Plans helpful object movements
- **Alignment Checker**: Validates action-goal alignment

All prompts are configurable in the `prompts/` directory.

## Development

### Project Structure

```
objectagentsdemo/
├── src/                    # Main source code
│   ├── main.py            # Application entry point
│   ├── agent_loop.py      # Main AI agent orchestration
│   ├── perceive.py        # Vision and activity analysis
│   ├── reason.py          # Goal inference
│   ├── act.py             # Action planning
│   ├── tablespace.py      # Object detection and tracking
│   ├── physical_object.py # Robotic object control
│   ├── display.py         # Visualization interface
│   └── utils/             # Utility functions
├── config/                # Scene configuration files
├── prompts/               # AI agent prompts
├── models/                # YOLO model files
└── fonts/                 # Display fonts
```

### Key Dependencies

- **ultralytics**: YOLO object detection
- **opencv-python**: Computer vision and display
- **openai**: LLM API access
- **bleak**: Bluetooth Low Energy communication
- **pydantic**: Data validation and parsing
- **tenacity**: Retry logic for API calls

### Adding New Objects

1. Update `config/scene_config.yaml` with object class mapping
2. Add object knowledge to `prompts/object_knowledge.yaml`
3. Train YOLO model to detect the new object (if needed)
4. For active objects, implement Bluetooth control protocol

## Troubleshooting

### Common Issues

- **Camera not found**: Check camera indices in `main.py`
- **Object detection failing**: Verify YOLO model paths and object classes
- **Bluetooth connection issues**: Ensure robotic objects are powered and in range
- **API errors**: Verify OpenAI API key and rate limits

### Debug Mode

Enable debugging by setting `agent_loop_debugging = True` in `main.py` to use video files instead of live cameras.
