# Idea Ring

Transform your brainstormed ideas into a structured timeline using artificial intelligence and smart optimization.

## What This Does

This tool takes a collection of ideas (like student feedback, brainstorming notes, or any text) and organizes them into a logical sequence arranged like a clock - from 12:01 AM to 11:59 AM. It's like having an AI assistant help you make sense of scattered thoughts by finding the most sensible order to present them.

## How It Works (No Math Background Needed!)

1. **Read Your Ideas**: The program reads ideas from a text file (one idea per line)
2. **Understand Meaning**: It uses AI to understand what each idea means (like how you'd understand the difference between "pizza" and "calculus")
3. **Find Connections**: It figures out which ideas are similar to each other
4. **Create the Best Path**: Like planning the most efficient route to visit all your friends' houses, it finds the best order to present your ideas
5. **Assign Times**: It spreads your ideas across a 12-hour clock, giving each one a timestamp

## Quick Start

### Prerequisites
- Python 3 installed on your computer
- Internet connection (for downloading AI models)

### Installation & Running

```bash
# Install required packages
make install

# Run with the default data file
make run

# Or run with your own file
python idea-ring.py -f your_file.txt
```

### Input Format

Create a text file where each line contains ideas separated by periods, commas, semicolons, or "and":

```
I want to learn about AI ethics, legal implications, and practical applications.
Machine learning in healthcare; deep learning fundamentals.
```

## Example Output

```
12:01 AM - AI ethics and user boundaries
12:30 AM - Legal implications of AI
01:00 AM - Applications of AI in science
01:30 AM - AI in medicine and healthcare
...
```

## What's Happening Behind the Scenes

- **Sentence Transformers**: Converts your text into numbers that represent meaning
- **Distance Calculation**: Measures how similar ideas are to each other
- **TSP Algorithm**: Solves the "Traveling Salesman Problem" to find the optimal path through all ideas
- **Time Mapping**: Distributes ideas evenly across a 12-hour period

## Files in This Project

- `idea-ring.py` - The main program
- `datafile.txt` - Example input file with student feedback about AI topics
- `requirements.txt` - List of needed Python packages
- `makefile` - Simple commands to install and run

## Customization

Change the input file by using the `-f` flag:
```bash
python idea-ring.py -f my_ideas.txt
```

## Why This is Useful

Perfect for:
- Organizing meeting notes
- Structuring presentation topics
- Sequencing curriculum content
- Planning project phases
- Making sense of survey responses

The "ring" concept helps create a natural flow through related topics, making complex information easier to understand and present.