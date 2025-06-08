# SA-GA Enhanced LoRA for Twitter Sentiment Analysis

## Project Overview

This project investigates the application of Low-Rank Adaptation (LoRA) for fine-tuning the LLaMA model on English Twitter sentiment classification tasks. To enhance model performance and tuning efficiency, we integrate Simulated Annealing (SA) and Genetic Algorithms (GA) for automatic hyperparameter optimization.

## Dataset

- Source: [Hugging Face - large-twitter-tweets-sentiment](https://huggingface.co/datasets/gxb912/large-twitter-tweets-sentiment)
- Description: A collection of English tweets annotated with sentiment labels.
- Structure:
  - `text`: Tweet content
  - `sentiment`: 1 for positive, 0 for negative

## Model and Approach

- Base model: LLaMA
- Fine-tuning technique: LoRA
- Training methods: Comparison of prompting and full fine-tuning
- Training framework: Hugging Face Transformers with Trainer API

## Training Pipeline

1. Task and dataset selection
2. Model and platform setup
3. Baseline model training
4. Prompting vs. fine-tuning comparison
5. Integration of optimization strategies (SA and GA)

## Optimization Methods

### Simulated Annealing (SA)

- Generates neighbor solutions and accepts based on Metropolis criterion
- Best configuration:
  - `r=32`, `alpha=15`, `dropout=0.082`, `learning_rate=7.35e-5`
  - F1 Score: 0.8441

### Genetic Algorithm Optimization (EvoLoRA)

- Representation: r, alpha, dropout, learning rate
- Fitness function: F1 score on validation set
- Operators:
  - Tournament selection
  - Two-point crossover
  - Gaussian mutation with adaptive probability
- Best configuration:
  - `r=14`, `alpha=32`, `dropout=0.12`, `learning_rate=3.5e-5`
  - F1 Score: 0.8427

## Experimental Results

| Method              | F1 Score | Best Configuration                                    |
|---------------------|----------|-------------------------------------------------------|
| Simulated Annealing | 0.8441   | r=32, alpha=15, dropout=0.082, lr=7.35e-5            |
| EvoLoRA (GA)        | 0.8427   | r=14, alpha=32, dropout=0.12, lr=3.5e-5              |

## Conclusion

- The project covers the full NLP pipeline from task definition to model evaluation.
- Compared prompting and fine-tuning approaches to analyze performance trade-offs.
- Integrated optimization methods significantly improved training efficiency and accuracy.

## Future Work

- Extend the dataset with more diverse or domain-specific samples
- Explore instruction tuning for improved alignment
- Evaluate robustness under challenging or adversarial scenarios
- Incorporate external knowledge (e.g., retrieval-augmented generation)
- Optimize for faster and more resource-efficient fine-tuning

## Team Members

Jiang Hu  
Yu Zhang  
Shiying Wei  
Zi Li  

Supervisor: Dr. John McCrae  
School of Computer Science
