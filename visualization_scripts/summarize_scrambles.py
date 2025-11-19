#!/usr/bin/env python3
"""
Script to summarize scrambled evaluation results across multiple seeds.
Takes a directory containing seed subdirectories and calculates averages
of ap50_instance_Aweighted and ap_instance_Aweighted metrics.
"""

import argparse
import os
import csv
import glob
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.table import Table


def find_seed_directories(base_dir):
    """Find all seed directories matching the pattern eval_*_seed*"""
    base_path = Path(base_dir)
    seed_dirs = []
    
    # Look for directories matching the pattern eval_*_seed*
    for item in base_path.iterdir():
        if item.is_dir() and 'seed' in item.name:
            seed_dirs.append(item)
    
    # Sort by seed number for consistent output
    seed_dirs.sort(key=lambda x: int(x.name.split('seed')[-1]) if x.name.split('seed')[-1].isdigit() else 999)
    
    return seed_dirs


def read_metrics_from_csv(csv_path):
    """Read ap50_instance_Aweighted and ap_instance_Aweighted from CSV file"""
    metrics = {}
    
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found: {csv_path}")
        return None
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_name = row['metric_name']
            if metric_name == 'ap50_instance_Aweighted':
                metrics['ap50_instance_Aweighted'] = float(row['value'])
            elif metric_name == 'ap_instance_Aweighted':
                metrics['ap_instance_Aweighted'] = float(row['value'])
    
    # Check if both metrics were found
    if 'ap50_instance_Aweighted' not in metrics or 'ap_instance_Aweighted' not in metrics:
        print(f"Warning: Missing metrics in {csv_path}")
        return None
    
    return metrics


def read_category_metrics_from_csv(csv_path):
    """Read per-category metrics from mask_metrics_category.csv file"""
    category_metrics = {}
    
    if not os.path.exists(csv_path):
        print(f"Warning: Category CSV file not found: {csv_path}")
        return None
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            category_name = row['category_name']
            ap_instance = row['ap_instance_per_cat']
            ap50_instance = row['ap50_instance_per_cat']
            
            # Handle empty values
            # Note: Category CSV values are percentages (0-100), keep as percentages
            if ap_instance and ap_instance.strip():
                category_metrics[category_name] = {
                    'ap_instance_per_cat': float(ap_instance),
                    'ap50_instance_per_cat': float(ap50_instance) if ap50_instance and ap50_instance.strip() else None
                }
            elif ap50_instance and ap50_instance.strip():
                category_metrics[category_name] = {
                    'ap_instance_per_cat': None,
                    'ap50_instance_per_cat': float(ap50_instance)
                }
    
    if not category_metrics:
        print(f"Warning: No category metrics found in {csv_path}")
        return None
    
    return category_metrics


def save_results_to_csv(csv_path, avg_ap50, sem_ap50, avg_ap, sem_ap, n_seeds, category_summary,
                        original_ap50=None, original_ap=None, original_category_summary=None):
    """Save summary results to CSV file"""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Combined table with dataset-level as "ALL" category
        if original_ap50 is not None:
            writer.writerow(['Category', 'Metric', 'Original', 'Scrambled Mean', 'Scrambled SEM', 'N'])
            
            # Write "ALL" category (dataset-level metrics)
            writer.writerow(['ALL', 'AP', f'{original_ap:.2f}', f'{avg_ap:.2f}', f'{sem_ap:.2f}', n_seeds])
            writer.writerow(['ALL', 'AP50', f'{original_ap50:.2f}', f'{avg_ap50:.2f}', f'{sem_ap50:.2f}', n_seeds])
            
            # Create a dict for quick lookup of original values
            original_dict = {cat['category']: cat for cat in original_category_summary} if original_category_summary else {}
            
            # Write per-category metrics
            for cat in category_summary:
                cat_name = cat['category']
                orig_cat = original_dict.get(cat_name, {})
                
                if cat['ap_instance_per_cat_mean'] is not None:
                    orig_ap = orig_cat.get('ap_instance_per_cat_mean', 'N/A')
                    orig_ap_str = f"{orig_ap:.2f}" if isinstance(orig_ap, (int, float)) else str(orig_ap)
                    writer.writerow([
                        cat_name,
                        'AP',
                        orig_ap_str,
                        f"{cat['ap_instance_per_cat_mean']:.2f}",
                        f"{cat['ap_instance_per_cat_sem']:.2f}",
                        cat['ap_instance_per_cat_n']
                    ])
                if cat['ap50_instance_per_cat_mean'] is not None:
                    orig_ap50 = orig_cat.get('ap50_instance_per_cat_mean', 'N/A')
                    orig_ap50_str = f"{orig_ap50:.2f}" if isinstance(orig_ap50, (int, float)) else str(orig_ap50)
                    writer.writerow([
                        cat_name,
                        'AP50',
                        orig_ap50_str,
                        f"{cat['ap50_instance_per_cat_mean']:.2f}",
                        f"{cat['ap50_instance_per_cat_sem']:.2f}",
                        cat['ap50_instance_per_cat_n']
                    ])
        else:
            writer.writerow(['Category', 'Metric', 'Mean', 'SEM', 'N'])
            # Write "ALL" category (dataset-level metrics)
            writer.writerow(['ALL', 'AP', f'{avg_ap:.2f}', f'{sem_ap:.2f}', n_seeds])
            writer.writerow(['ALL', 'AP50', f'{avg_ap50:.2f}', f'{sem_ap50:.2f}', n_seeds])
            
            # Write per-category metrics
            for cat in category_summary:
                if cat['ap_instance_per_cat_mean'] is not None:
                    writer.writerow([
                        cat['category'],
                        'AP',
                        f"{cat['ap_instance_per_cat_mean']:.2f}",
                        f"{cat['ap_instance_per_cat_sem']:.2f}",
                        cat['ap_instance_per_cat_n']
                    ])
                if cat['ap50_instance_per_cat_mean'] is not None:
                    writer.writerow([
                        cat['category'],
                        'AP50',
                        f"{cat['ap50_instance_per_cat_mean']:.2f}",
                        f"{cat['ap50_instance_per_cat_sem']:.2f}",
                        cat['ap50_instance_per_cat_n']
                    ])


def create_summary_table_png(png_path, avg_ap50, sem_ap50, avg_ap, sem_ap, n_seeds, category_summary,
                             original_ap50=None, original_ap=None, original_category_summary=None):
    """Create a PNG table visualization of the summary results"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data - combined table with "ALL" category
    table_data = []
    
    # Header
    if original_ap50 is not None:
        table_data.append(['Evaluation Summary', '', '', '', '', '', ''])
        table_data.append(['Category', 'Original AP', 'Scrambled AP Mean', 'Scrambled AP SEM', 
                          'Original AP50', 'Scrambled AP50 Mean', 'Scrambled AP50 SEM'])
        
        # "ALL" category (dataset-level metrics) - will be bolded
        table_data.append([
            'ALL',
            f'{original_ap:.2f}',
            f'{avg_ap:.2f}',
            f'{sem_ap:.2f}',
            f'{original_ap50:.2f}',
            f'{avg_ap50:.2f}',
            f'{sem_ap50:.2f}'
        ])
        
        # Per-category metrics
        original_dict = {cat['category']: cat for cat in original_category_summary} if original_category_summary else {}
        
        for cat in category_summary:
            cat_name = cat['category']
            orig_cat = original_dict.get(cat_name, {})
            
            orig_ap = orig_cat.get('ap_instance_per_cat_mean', None)
            orig_ap_str = f"{orig_ap:.2f}" if isinstance(orig_ap, (int, float)) else 'N/A'
            ap_mean = f"{cat['ap_instance_per_cat_mean']:.2f}" if cat['ap_instance_per_cat_mean'] is not None else 'N/A'
            ap_sem = f"{cat['ap_instance_per_cat_sem']:.2f}" if cat['ap_instance_per_cat_sem'] is not None else 'N/A'
            
            orig_ap50 = orig_cat.get('ap50_instance_per_cat_mean', None)
            orig_ap50_str = f"{orig_ap50:.2f}" if isinstance(orig_ap50, (int, float)) else 'N/A'
            ap50_mean = f"{cat['ap50_instance_per_cat_mean']:.2f}" if cat['ap50_instance_per_cat_mean'] is not None else 'N/A'
            ap50_sem = f"{cat['ap50_instance_per_cat_sem']:.2f}" if cat['ap50_instance_per_cat_sem'] is not None else 'N/A'
            
            table_data.append([
                cat_name,
                orig_ap_str,
                ap_mean,
                ap_sem,
                orig_ap50_str,
                ap50_mean,
                ap50_sem
            ])
    else:
        table_data.append(['Evaluation Summary', '', '', '', ''])
        table_data.append(['Category', 'AP Mean', 'AP SEM', 'AP50 Mean', 'AP50 SEM'])
        
        # "ALL" category (dataset-level metrics) - will be bolded
        table_data.append([
            'ALL',
            f'{avg_ap:.2f}',
            f'{sem_ap:.2f}',
            f'{avg_ap50:.2f}',
            f'{sem_ap50:.2f}'
        ])
        
        # Per-category metrics
        for cat in category_summary:
            ap_mean = f"{cat['ap_instance_per_cat_mean']:.2f}" if cat['ap_instance_per_cat_mean'] is not None else 'N/A'
            ap_sem = f"{cat['ap_instance_per_cat_sem']:.2f}" if cat['ap_instance_per_cat_sem'] is not None else 'N/A'
            ap50_mean = f"{cat['ap50_instance_per_cat_mean']:.2f}" if cat['ap50_instance_per_cat_mean'] is not None else 'N/A'
            ap50_sem = f"{cat['ap50_instance_per_cat_sem']:.2f}" if cat['ap50_instance_per_cat_sem'] is not None else 'N/A'
            
            table_data.append([
                cat['category'],
                ap_mean,
                ap_sem,
                ap50_mean,
                ap50_sem
            ])
    
    # Create table - ensure all rows have the same number of columns
    if table_data:
        num_cols = max(len(row) for row in table_data)
        # Pad all rows to have the same number of columns
        for i, row in enumerate(table_data):
            while len(row) < num_cols:
                row.append('')
        
        col_widths = [0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12] if original_category_summary else [0.25, 0.15, 0.15, 0.15, 0.15, 0.05]
        table = ax.table(cellText=table_data, cellLoc='center', loc='center', 
                         colWidths=col_widths[:num_cols])
    else:
        return
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header rows and "ALL" row
    for i in range(len(table_data)):
        row = table_data[i]
        if len(row) > 0 and row[0] == 'Evaluation Summary':  # Section header
            for j in range(num_cols):
                table[(i, j)].set_facecolor('#4472C4')
                table[(i, j)].set_text_props(weight='bold', color='white')
        elif len(row) > 0 and row[0] == 'Category':  # Column headers
            for j in range(len(row)):
                table[(i, j)].set_facecolor('#D9E1F2')
                table[(i, j)].set_text_props(weight='bold')
        elif len(row) > 0 and row[0] == 'ALL':  # "ALL" row - make it bold
            for j in range(len(row)):
                table[(i, j)].set_text_props(weight='bold')
        elif len(row) > 0 and all(cell == '' for cell in row):  # Empty row
            for j in range(num_cols):
                table[(i, j)].set_facecolor('#FFFFFF')
                table[(i, j)].set_height(0.1)
    
    plt.title('Scrambled Evaluation Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()


def extract_checkpoint_number(seed_dir_name):
    """Extract checkpoint number from seed directory name (e.g., eval_6059_all_frames_seed1 -> 6059)"""
    import re
    # Look for pattern like eval_6059 or eval_6059_all_frames
    match = re.search(r'eval_(\d+)', seed_dir_name)
    if match:
        return match.group(1)
    return None


def read_original_checkpoint_metrics(base_dir, checkpoint_num):
    """Read original checkpoint metrics from checkpoint_evaluations directory"""
    if checkpoint_num is None:
        return None, None
    
    # Construct checkpoint directory path (e.g., checkpoint_0006059)
    checkpoint_dir = f"checkpoint_{checkpoint_num:07d}"
    # base_dir is like .../model_camera_fold4/scrambled, so go up one level to model_camera_fold4
    checkpoint_eval_dir = os.path.join(os.path.dirname(base_dir), 
                                      'checkpoint_evaluations', checkpoint_dir)
    
    if not os.path.isdir(checkpoint_eval_dir):
        print(f"Warning: Checkpoint evaluation directory not found: {checkpoint_eval_dir}")
        return None, None
    
    # Read dataset metrics
    dataset_csv = os.path.join(checkpoint_eval_dir, 'inference', 'mask_metrics_dataset.csv')
    dataset_metrics = read_metrics_from_csv(dataset_csv)
    
    # Read category metrics
    category_csv = os.path.join(checkpoint_eval_dir, 'inference', 'mask_metrics_category.csv')
    category_metrics = read_category_metrics_from_csv(category_csv)
    
    return dataset_metrics, category_metrics


def summarize_scrambles(base_dir):
    """Main function to summarize scrambled evaluation results"""
    base_dir = os.path.abspath(base_dir)
    
    if not os.path.isdir(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        return
    
    # Find all seed directories
    seed_dirs = find_seed_directories(base_dir)
    
    if not seed_dirs:
        print(f"Error: No seed directories found in {base_dir}")
        return
    
    print(f"Found {len(seed_dirs)} seed directories in {base_dir}\n")
    
    # Extract checkpoint number from first seed directory
    checkpoint_num = None
    if seed_dirs:
        checkpoint_num_str = extract_checkpoint_number(seed_dirs[0].name)
        if checkpoint_num_str:
            checkpoint_num = int(checkpoint_num_str)
            print(f"Detected checkpoint number: {checkpoint_num}\n")
    
    # Read original checkpoint metrics
    original_dataset_metrics, original_category_metrics = read_original_checkpoint_metrics(base_dir, checkpoint_num)
    
    # Collect metrics from each seed directory
    all_ap50_values = []
    all_ap_values = []
    successful_seeds = []
    
    # Dictionary to store per-category metrics: {category_name: {'ap': [values], 'ap50': [values]}}
    category_data = {}
    
    for seed_dir in seed_dirs:
        csv_path = os.path.join(seed_dir, 'inference', 'mask_metrics_dataset.csv')
        metrics = read_metrics_from_csv(csv_path)
        
        if metrics is not None:
            all_ap50_values.append(metrics['ap50_instance_Aweighted'])
            all_ap_values.append(metrics['ap_instance_Aweighted'])
            successful_seeds.append(seed_dir.name)
            print(f"{seed_dir.name}:")
            print(f"  ap50_instance_Aweighted: {metrics['ap50_instance_Aweighted'] * 100.0:.2f}")
            print(f"  ap_instance_Aweighted: {metrics['ap_instance_Aweighted'] * 100.0:.2f}")
            
            # Read category metrics
            category_csv_path = os.path.join(seed_dir, 'inference', 'mask_metrics_category.csv')
            cat_metrics = read_category_metrics_from_csv(category_csv_path)
            
            if cat_metrics is not None:
                for category_name, values in cat_metrics.items():
                    if category_name not in category_data:
                        category_data[category_name] = {'ap': [], 'ap50': []}
                    
                    if values['ap_instance_per_cat'] is not None:
                        category_data[category_name]['ap'].append(values['ap_instance_per_cat'])
                    if values['ap50_instance_per_cat'] is not None:
                        category_data[category_name]['ap50'].append(values['ap50_instance_per_cat'])
        else:
            print(f"{seed_dir.name}: Failed to read metrics")
    
    if not all_ap50_values:
        print("\nError: No valid metrics found in any seed directory")
        return
    
    # Calculate averages
    n = len(all_ap50_values)
    avg_ap50 = sum(all_ap50_values) / n
    avg_ap = sum(all_ap_values) / n
    
    # Calculate standard deviations
    def std_dev(values, mean):
        """Calculate sample standard deviation"""
        if len(values) <= 1:
            return 0.0
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    std_ap50 = std_dev(all_ap50_values, avg_ap50)
    std_ap = std_dev(all_ap_values, avg_ap)
    
    # Calculate standard error of the mean
    sem_ap50 = std_ap50 / math.sqrt(n)
    sem_ap = std_ap / math.sqrt(n)
    
    # Convert dataset-level metrics from decimals (0-1) to percentages (0-100)
    avg_ap50_pct = avg_ap50 * 100.0
    avg_ap_pct = avg_ap * 100.0
    sem_ap50_pct = sem_ap50 * 100.0
    sem_ap_pct = sem_ap * 100.0
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Number of seeds processed: {len(successful_seeds)}")
    print(f"\nAverage ap50_instance_Aweighted: {avg_ap50_pct:.2f} ± {sem_ap50_pct:.2f}")
    print(f"Average ap_instance_Aweighted: {avg_ap_pct:.2f} ± {sem_ap_pct:.2f}")
    
    # Calculate and print per-category metrics
    category_summary = []
    if category_data:
        print("\nPer-Category Metrics:")
        print("-" * 60)
        
        # Sort categories by name for consistent output
        sorted_categories = sorted(category_data.keys())
        
        for category_name in sorted_categories:
            cat_ap_values = category_data[category_name]['ap']
            cat_ap50_values = category_data[category_name]['ap50']
            
            cat_summary = {'category': category_name}
            
            if cat_ap_values:
                n_cat = len(cat_ap_values)
                avg_cat_ap = sum(cat_ap_values) / n_cat
                std_cat_ap = std_dev(cat_ap_values, avg_cat_ap)
                sem_cat_ap = std_cat_ap / math.sqrt(n_cat) if n_cat > 1 else 0.0
                print(f"{category_name}:")
                print(f"  ap_instance_per_cat: {avg_cat_ap:.2f} ± {sem_cat_ap:.2f} (n={n_cat})")
                cat_summary['ap_instance_per_cat_mean'] = avg_cat_ap
                cat_summary['ap_instance_per_cat_sem'] = sem_cat_ap
                cat_summary['ap_instance_per_cat_n'] = n_cat
            else:
                cat_summary['ap_instance_per_cat_mean'] = None
                cat_summary['ap_instance_per_cat_sem'] = None
                cat_summary['ap_instance_per_cat_n'] = 0
            
            if cat_ap50_values:
                n_cat50 = len(cat_ap50_values)
                avg_cat_ap50 = sum(cat_ap50_values) / n_cat50
                std_cat_ap50 = std_dev(cat_ap50_values, avg_cat_ap50)
                sem_cat_ap50 = std_cat_ap50 / math.sqrt(n_cat50) if n_cat50 > 1 else 0.0
                if not cat_ap_values:  # Only print category name if we didn't print it above
                    print(f"{category_name}:")
                print(f"  ap50_instance_per_cat: {avg_cat_ap50:.2f} ± {sem_cat_ap50:.2f} (n={n_cat50})")
                cat_summary['ap50_instance_per_cat_mean'] = avg_cat_ap50
                cat_summary['ap50_instance_per_cat_sem'] = sem_cat_ap50
                cat_summary['ap50_instance_per_cat_n'] = n_cat50
            else:
                cat_summary['ap50_instance_per_cat_mean'] = None
                cat_summary['ap50_instance_per_cat_sem'] = None
                cat_summary['ap50_instance_per_cat_n'] = 0
            
            category_summary.append(cat_summary)
    
    print("="*60)
    
    # Prepare original metrics for comparison
    original_ap50_pct = None
    original_ap_pct = None
    original_category_summary = []
    
    if original_dataset_metrics:
        original_ap50_pct = original_dataset_metrics['ap50_instance_Aweighted'] * 100.0
        original_ap_pct = original_dataset_metrics['ap_instance_Aweighted'] * 100.0
        print(f"\nOriginal checkpoint metrics:")
        print(f"  ap50_instance_Aweighted: {original_ap50_pct:.2f}")
        print(f"  ap_instance_Aweighted: {original_ap_pct:.2f}")
    
    if original_category_metrics:
        sorted_original_cats = sorted(original_category_metrics.keys())
        for cat_name in sorted_original_cats:
            cat_data = original_category_metrics[cat_name]
            original_category_summary.append({
                'category': cat_name,
                'ap_instance_per_cat_mean': cat_data['ap_instance_per_cat'],
                'ap_instance_per_cat_sem': None,  # Original has no SEM
                'ap50_instance_per_cat_mean': cat_data['ap50_instance_per_cat'],
                'ap50_instance_per_cat_sem': None  # Original has no SEM
            })
    
    # Save results to CSV
    csv_path = os.path.join(base_dir, 'scrambled_summary.csv')
    save_results_to_csv(csv_path, avg_ap50_pct, sem_ap50_pct, avg_ap_pct, sem_ap_pct, 
                       len(successful_seeds), category_summary,
                       original_ap50_pct, original_ap_pct, original_category_summary)
    print(f"\nResults saved to: {csv_path}")
    
    # Create and save PNG table
    png_path = os.path.join(base_dir, 'scrambled_summary.png')
    create_summary_table_png(png_path, avg_ap50_pct, sem_ap50_pct, avg_ap_pct, sem_ap_pct,
                            len(successful_seeds), category_summary,
                            original_ap50_pct, original_ap_pct, original_category_summary)
    print(f"Table saved to: {png_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Summarize scrambled evaluation results across multiple seeds'
    )
    parser.add_argument(
        'directory',
        type=str,
        help='Base directory containing seed subdirectories (e.g., /path/to/scrambled)'
    )
    
    args = parser.parse_args()
    summarize_scrambles(args.directory)


if __name__ == '__main__':
    main()

