from text_extraction import process_manifestos
from transformers import pipeline
import torch
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

INPUT_DIR = "data/pdfs"
PROCESSED_DIR = "data/extracted_text"
RESULTS_DIR = "results"


def sentiment_analysis(bin_pipe: pipeline, multi_pipe: pipeline) -> pd.DataFrame:
    ''' Perform sentiment analysis on processed text files and save results to CSV '''

    files = [f for f in os.listdir(PROCESSED_DIR) if f.lower().endswith('.txt')]
    
    for f in files:
        results = []
        print(f"Analyzing sentiment for: {f}")
        with open(os.path.join(PROCESSED_DIR, f), 'r', encoding='utf-8') as file:
            for line in file:
                # skip page markers and page numbers
                if line.strip() and not line.startswith("---") and not line.isdigit(): 
                    bin_result = bin_pipe(line.strip())
                    multi_result = multi_pipe(line.strip())
                    
                    combined_result = {
                        'text': line[:128].strip(),
                        'binary_label': bin_result[0]['label'],
                        'binary_score': bin_result[0]['score'],
                        'multi_label': multi_result[0]['label'],
                        'multi_score': multi_result[0]['score']
                    }
                    
                    results.append(combined_result)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(RESULTS_DIR, f.replace('.txt', '_sentiment.csv')), index=False)

    return results_df

def get_statistics_per_party(df: pd.DataFrame, party: str) -> dict:
    ''' Generate summary statistics from sentiment analysis results per party'''
    
    summary = {
        'party': party,
        'total_entries': len(df),
        'binary_pos': len(df[df['binary_label'] == 'pos']),
        'bin_pos%': (len(df[df['binary_label'] == 'pos']) / len(df) * 100) if len(df) > 0 else 0,
        'avg_bin_pos_score': df[df['binary_label'] == 'pos']['binary_score'].mean() if len(df[df['binary_label'] == 'pos']) > 0 else 0,
        'binary_neg': len(df[df['binary_label'] == 'neg']),
        'bin_neg%': (len(df[df['binary_label'] == 'neg']) / len(df) * 100) if len(df) > 0 else 0,
        'avg_bin_neg_score': df[df['binary_label'] == 'neg']['binary_score'].mean() if len(df[df['binary_label'] == 'neg']) > 0 else 0,
        'avg_bin_score': df['binary_score'].mean() if len(df) > 0 else 0,
        'multi_very_pos': len(df[df['multi_label'] == 'very pos']),
        'multi_very_pos%': (len(df[df['multi_label'] == 'very pos']) / len(df) * 100) if len(df) > 0 else 0,
        'avg_multi_very_pos_score': df[df['multi_label'] == 'very pos']['multi_score'].mean() if len(df[df['multi_label'] == 'very pos']) > 0 else 0,
        'multi_pos': len(df[df['multi_label'] == 'pos']),
        'multi_pos%': (len(df[df['multi_label'] == 'pos']) / len(df) * 100) if len(df) > 0 else 0,
        'avg_multi_pos_score': df[df['multi_label'] == 'pos']['multi_score'].mean() if len(df[df['multi_label'] == 'pos']) > 0 else 0,
        'multi_neut': len(df[df['multi_label'] == 'neut']),
        'multi_neut%': (len(df[df['multi_label'] == 'neut']) / len(df) * 100) if len(df) > 0 else 0,
        'avg_multi_neut_score': df[df['multi_label'] == 'neut']['multi_score'].mean() if len(df[df['multi_label'] == 'neut']) > 0 else 0,
        'multi_neg': len(df[df['multi_label'] == 'neg']),
        'multi_neg%': (len(df[df['multi_label'] == 'neg']) / len(df) * 100) if len(df) > 0 else 0,
        'avg_multi_neg_score': df[df['multi_label'] == 'neg']['multi_score'].mean() if len(df[df['multi_label'] == 'neg']) > 0 else 0,
        'multi_very_neg': len(df[df['multi_label'] == 'very neg']),
        'multi_very_neg%': (len(df[df['multi_label'] == 'very neg']) / len(df) * 100) if len(df) > 0 else 0,
        'avg_multi_very_neg_score': df[df['multi_label'] == 'very neg']['multi_score'].mean() if len(df[df['multi_label'] == 'very neg']) > 0 else 0,
        'avg_multi_score': df['multi_score'].mean() if len(df) > 0 else 0,
    }

    return summary

def get_overall_statistics() -> pd.DataFrame:
    ''' Compile overall statistics from all sentiment analysis results '''
    
    summary_list_per_party = []
    summary_list_per_group = []
    result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('_sentiment.csv')]

    left_wing_parties = ['pvdd', 'denk', 'sp', 'volt', 'gl-pvda']
    left_wing_df = pd.DataFrame()
    central_parties = ['cu', 'd66', '50plus', 'cda'] 
    central_df = pd.DataFrame()
    right_wing_parties = ['vvd', 'bbb', 'fvd', 'sgp', 'pvv', 'ja21']  
    right_wing_df = pd.DataFrame()

    progressive_parties = ['pvdd', 'sp', 'volt', 'gl-pvda', 'd66', 'denk']
    progressive_df = pd.DataFrame()
    conservative_parties = ['vvd', 'bbb', 'fvd', 'sgp', 'pvv', 'ja21']
    conservative_df = pd.DataFrame()
    religious_parties = ['cu', 'cda', 'sgp']
    religious_df = pd.DataFrame()

    
    for f in result_files:
        party_name = f.replace('_sentiment.csv', '')
        df = pd.read_csv(os.path.join(RESULTS_DIR, f))
        if party_name in left_wing_parties:
            left_wing_df = pd.concat([left_wing_df, df], ignore_index=True)
        elif party_name in central_parties:
            central_df = pd.concat([central_df, df], ignore_index=True)
        elif party_name in right_wing_parties:
            right_wing_df = pd.concat([right_wing_df, df], ignore_index=True)

        if party_name in progressive_parties:
            progressive_df = pd.concat([progressive_df, df], ignore_index=True)
        elif party_name in conservative_parties:
            conservative_df = pd.concat([conservative_df, df], ignore_index=True)
        
        if party_name in religious_parties:
            religious_df = pd.concat([religious_df, df], ignore_index=True)

        party_summary = get_statistics_per_party(df, party_name)
        summary_list_per_party.append(party_summary)

    if not left_wing_df.empty:
        left_wing_summary = get_statistics_per_party(left_wing_df, 'Left Wing Aggregate')
        summary_list_per_group.append(left_wing_summary)
    if not central_df.empty:
        central_summary = get_statistics_per_party(central_df, 'Central Aggregate')
        summary_list_per_group.append(central_summary)
    if not right_wing_df.empty:
        right_wing_summary = get_statistics_per_party(right_wing_df, 'Right Wing Aggregate')
        summary_list_per_group.append(right_wing_summary)

    if not progressive_df.empty:
        progressive_summary = get_statistics_per_party(progressive_df, 'Progressive Aggregate')
        summary_list_per_group.append(progressive_summary)
    if not conservative_df.empty:
        conservative_summary = get_statistics_per_party(conservative_df, 'Conservative Aggregate')
        summary_list_per_group.append(conservative_summary)

    if not religious_df.empty:
        religious_summary = get_statistics_per_party(religious_df, 'Religious Aggregate')
        summary_list_per_group.append(religious_summary)
    
    seats = {
        'd66': 26,
        'pvv': 26,
        'vvd': 22,
        'gl-pvda': 20,
        'cda': 18,
        'ja21': 9,
        'fvd': 7,
        'bbb': 4,
        'sp': 3,
        'denk': 3,
        'cu': 3,
        'pvdd': 3,
        'sgp': 3,
        '50plus': 2,
        'volt': 1
    }

    summary_per_party_df = pd.DataFrame(summary_list_per_party)
    summary_per_party_df['seats'] = summary_per_party_df['party'].map(seats).fillna(0).astype(int)
    summary_per_party_df = summary_per_party_df.sort_values(by='seats', ascending=False)

    summary_per_group_df = pd.DataFrame(summary_list_per_group)
    summary_per_party_df.to_csv(os.path.join(RESULTS_DIR, 'sentiment_summary_per_party.csv'), index=False)
    summary_per_group_df.to_csv(os.path.join(RESULTS_DIR, 'sentiment_summary_per_group.csv'), index=False)
    
    return summary_per_party_df, summary_per_group_df

def plot_statistics(stats_df: pd.DataFrame, name: str='party'):
    ''' Plot sentiment statistics for visualization '''
    
    plt.figure(figsize=(12, 6))
    plt.bar(stats_df['party'], stats_df['bin_pos%'], color='green', alpha=0.6, label='Binary Positive %')
    plt.bar(stats_df['party'], stats_df['bin_neg%'], bottom=stats_df['bin_pos%'], color='red', alpha=0.6, label='Binary Negative %')
    plt.xlabel('Political Party')
    plt.ylabel('Percentage (%)')
    plt.title('Sentiment Analysis Summary by Party')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'binary_sentiment_summary_{name}_plot.png'))
    plt.figure(figsize=(12, 6))
    plt.bar(stats_df['party'], stats_df['multi_very_pos%'], color='darkgreen', alpha=0.6, label='Multi Very Positive %')
    plt.bar(stats_df['party'], stats_df['multi_pos%'], bottom=stats_df['multi_very_pos%'], color='lightgreen', alpha=0.6, label='Multi Positive %')
    bottom_neut = stats_df['multi_very_pos%'] + stats_df['multi_pos%']
    plt.bar(stats_df['party'], stats_df['multi_neut%'], bottom=bottom_neut, color='gray', alpha=0.6, label='Multi Neutral %')
    bottom_neg = bottom_neut + stats_df['multi_neut%']
    plt.bar(stats_df['party'], stats_df['multi_neg%'], bottom=bottom_neg, color='orange', alpha=0.6, label='Multi Negative %')
    bottom_very_neg = bottom_neg + stats_df['multi_neg%']
    plt.bar(stats_df['party'], stats_df['multi_very_neg%'], bottom=bottom_very_neg, color='darkred', alpha=0.6, label='Multi Very Negative %')
    plt.xlabel('Political Party')
    plt.ylabel('Percentage (%)')
    plt.title('Multi-label Sentiment Analysis Summary by Party')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'multi_sentiment_summary_{name}_plot.png'))
    plt.show()

def main():
    # process_manifestos(INPUT_DIR, PROCESSED_DIR)
    # device = "mps" if torch.backends.mps.is_available() else "cpu"  # or "cpu" if no GPU is available

    # # Use a pipeline as a high-level helper
    # multi_label_pipe = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment", device=device)
    # multi_label_pipe.model.config.id2label = {0: "very neg", 1: "neg", 2: "neut", 3: "pos", 4: "very pos"}
    # multi_label_pipe.model.config.label2id = {"very neg": 0, "neg": 1, "neut": 2, "pos": 3, "very pos": 4}

    # binary_pipe = pipeline("text-classification", model="wietsedv/bert-base-dutch-cased-finetuned-sentiment", device=device)

    # sentiment_analysis(binary_pipe, multi_label_pipe)
    
    stats_per_party, stats_per_group = get_overall_statistics()

    plot_statistics(stats_per_party, name='party')
    plot_statistics(stats_per_group, name='group')

if __name__ == "__main__":
    main()