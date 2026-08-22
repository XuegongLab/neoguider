import os
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
import re

def extract_peptides_from_user_file(user_file_path):
    """
    从用户结果文件中提取ET_pep列的肽段序列
    """
    try:
        # 读取用户文件（TSV格式）
        df = pd.read_csv(user_file_path, sep='\t', header=0)
        
        # 检查是否存在ET_pep列
        if 'ET_pep' not in df.columns:
            raise ValueError("用户文件中未找到'ET_pep'列")
        
        # 提取肽段序列，去重并过滤空值
        user_peptides = df['ET_pep'].dropna().unique().tolist()
        # 过滤无效肽段（长度过短或包含非氨基酸字符）
        user_peptides = [pep for pep in user_peptides if is_valid_peptide(pep)]
        
        print(f"从用户文件中提取到 {len(user_peptides)} 个唯一有效肽段")
        return user_peptides
    
    except Exception as e:
        print(f"读取用户文件失败: {e}")
        return []

def is_valid_peptide(peptide):
    """
    验证肽段序列的有效性（仅包含标准氨基酸代码，长度合理）
    标准氨基酸代码：A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
    """
    if not isinstance(peptide, str):
        return False
    # 肽段长度通常在8-15个氨基酸之间
    if len(peptide) < 6 or len(peptide) > 20:
        return False
    # 仅包含允许的氨基酸字符（不区分大小写）
    valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
    return all(c.upper() in valid_chars for c in peptide)

#import pandas as pd
from pathlib import Path
#import zipfile
from typing import Optional

def read_file_to_df(filepath: str) -> pd.DataFrame:
    """
    Reads a file (CSV/TSV/XLSX/ZIP/BZIP/B2Z) and returns a pandas DataFrame.
    
    Args:
        filepath: Path to the input file (ends with .csv, .tsv, .xlsx, .zip, .bzip, .b2z)
    
    Returns:
        pandas DataFrame containing the file data
    
    Raises:
        FileNotFoundError: If the input file does not exist
        ValueError: If the file extension is unsupported or zip file is invalid
        pd.errors.EmptyDataError: If the file is empty
        Exception: For other reading/parsing errors
    """
    # Convert to Path object for consistent path handling
    file_path = Path(filepath).resolve()
    
    # 1. Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found at: {file_path}")
    
    # 2. Get lowercase file extension (handles case insensitivity: .CSV, .Tsv, etc.)
    file_ext = file_path.suffix.lower()
    supported_exts = ['.csv', '.tsv', '.xlsx', '.bzip', '.b2z']
    
    # 3. Read file based on extension
    try:
        if file_ext == '.csv':
            # Explicit comma separator for CSV
            return pd.read_csv(file_path, encoding='utf-8', sep=',')
        
        elif file_ext == '.tsv':
            # Explicit tab separator for TSV
            return pd.read_csv(file_path, encoding='utf-8', sep='\t')
        
        elif file_ext == '.xlsx':
            # Read Excel file (supports .xlsx only; use openpyxl engine)
            return pd.read_excel(file_path, engine='openpyxl')
        
        elif file_ext in ['.bzip', '.b2z']:
            # Handle BZIP/B2Z compressed files (auto-detect CSV/TSV via separator inference)
            return pd.read_csv(
                file_path,
                encoding='utf-8',
                sep='infer',  # Auto-detect comma/tab
                compression='bz2'  # Both .bzip and .b2z map to bz2 compression
            )
        else:
            warnings.warn(
                f"Unsupported file extension: {file_path.suffix}\n"
                f"Supported extensions: {', '.join(supported_exts)}\n"
                f"Default to TSV file without header"
            )
            return pd.read_csv(
                file_path,
                encoding='utf-8',
                sep='infer',  # Auto-detect comma/tab
                header=None,
                index_col=None
            )
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"File is empty: {file_path}")
    except Exception as e:
        raise Exception(f"Failed to read file {file_path}: {str(e)}")

def extract_peptides_from_training_files(training_files):
    """
    从所有训练数据文件中提取肽段序列
    处理不同格式的文件：空格分隔、制表符分隔、逗号分隔等
    """
    all_training_peptides = set()
    
    for file_path in training_files:
        try:
            print(f"\n正在处理训练文件: {os.path.basename(file_path)}")
            
            # 读取文件内容
            df = read_file_to_df(file_path)
            pep_colname = ''
            for colname in [
                    'Peptide', # MixMHCpred and MHC-motif-atlas
                    'peptide', # MHCflurry
                    ]:
                if colname in df.columns:
                    pep_colname = colname
            if pep_colname == '':
                file_peptides = list(df.iloc[:,0])
            else:
                file_peptides = list(df.loc[:,pep_colname])
            
            # 去重并添加到总集合
            unique_peptides = set(file_peptides)
            all_training_peptides.update(unique_peptides)
            
            print(f"  - 提取到 {len(unique_peptides)} 个唯一有效肽段")
            
        except Exception as e:
            print(f"  ! 处理文件 {os.path.basename(file_path)} 失败: {e}")
            continue
    
    print(f"\n所有训练文件共提取到 {len(all_training_peptides)} 个唯一有效肽段")
    return list(all_training_peptides)

def find_leaked_peptides(user_peptides, training_peptides):
    """
    查找用户肽段中存在于训练数据中的肽段（可能的泄露）
    """
    # 统一转为大写进行比较
    user_peptides_upper = [pep.upper() for pep in user_peptides]
    training_peptides_upper = set([pep.upper() for pep in training_peptides])
    
    # 查找交集（泄露的肽段）
    leaked_peptides = [pep for pep in user_peptides_upper if pep in training_peptides_upper]
    unique_leaked = list(set(leaked_peptides))  # 去重
    
    # 唯一的肽段（未泄露）
    unique_user_peptides = list(set(user_peptides_upper))
    non_leaked_peptides = [pep for pep in unique_user_peptides if pep not in training_peptides_upper]
    
    return {
        'total_user_peptides': len(unique_user_peptides),
        'total_training_peptides': len(training_peptides_upper),
        'leaked_peptides': unique_leaked,
        'num_leaked': len(unique_leaked),
        'non_leaked_peptides': non_leaked_peptides,
        'num_non_leaked': len(non_leaked_peptides),
        'leakage_rate': len(unique_leaked) / len(unique_user_peptides) * 100 if unique_user_peptides else 0
    }

def generate_report(leakage_result, output_dir='leakage_report'):
    """
    生成详细的泄露检测报告
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成报告文件
    report_path = os.path.join(output_dir, 'data_leakage_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("                pMHC 数据泄露检测报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"检测时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. 数据概况\n")
        f.write("-" * 30 + "\n")
        f.write(f"用户软件生成的唯一肽段总数: {leakage_result['total_user_peptides']}\n")
        f.write(f"第三方训练数据中的唯一肽段总数: {leakage_result['total_training_peptides']}\n")
        f.write(f"重复肽段数量（可能泄露）: {leakage_result['num_leaked']}\n")
        f.write(f"唯一肽段数量（未泄露）: {leakage_result['num_non_leaked']}\n")
        f.write(f"数据泄露率: {leakage_result['leakage_rate']:.2f}%\n\n")
        
        f.write("2. 可能泄露的肽段列表\n")
        f.write("-" * 30 + "\n")
        if leakage_result['leaked_peptides']:
            for i, pep in enumerate(leakage_result['leaked_peptides'], 1):
                f.write(f"{i:3d}. {pep}\n")
        else:
            f.write("未发现泄露的肽段\n")
        f.write("\n")
        
        f.write("3. 未泄露的肽段列表\n")
        f.write("-" * 30 + "\n")
        if leakage_result['non_leaked_peptides']:
            for i, pep in enumerate(leakage_result['non_leaked_peptides'], 1):
                f.write(f"{i:3d}. {pep}\n")
        else:
            f.write("所有肽段均存在于训练数据中，泄露风险极高\n")
    
    print(f"\n报告已生成: {report_path}")
    
    # 生成CSV格式的详细结果（便于进一步分析）
    csv_path = os.path.join(output_dir, 'leakage_details.csv')
    details_data = []
    
    # 添加泄露的肽段
    for pep in leakage_result['leaked_peptides']:
        details_data.append({
            '肽段序列': pep,
            '状态': '可能泄露',
            '备注': '该肽段存在于第三方训练数据中'
        })
    
    # 添加未泄露的肽段
    for pep in leakage_result['non_leaked_peptides']:
        details_data.append({
            '肽段序列': pep,
            '状态': '未泄露',
            '备注': '该肽段仅存在于用户数据中'
        })
    
    pd.DataFrame(details_data).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"详细结果已保存为CSV: {csv_path}")
    
    return report_path, csv_path

def main():
    """
    主函数：执行完整的泄露检测流程
    """
    print("=" * 60)
    print("                pMHC 数据泄露检测工具")
    print("=" * 60 + "\n")
    
    # 1. 配置文件路径
    # 用户结果文件（请根据实际路径修改）
    USER_FILE = "/mnt/d/heteroclitic/v07-hpep.outdir/prioritization/v07_prelim_features_from_pmhcs.tsv.expansion.untraced" # "v07_prelim_features_from_pmhcs.tsv.expansion.untraced"
    
    # 训练数据文件（所有其他文件，排除用户自己的结果文件）
    TRAINING_FILES = [
        #"allelelist.head.txt",
        #"allelelist.head.txt.head.txt",
        "NetMHCpan_train/c000_ba",
        "NetMHCpan_train/c001_ba",
        "NetMHCpan_train/c002_ba",
        "NetMHCpan_train/c003_ba",
        "NetMHCpan_train/c004_ba",
        "NetMHCpan_train/c000_el",
        "NetMHCpan_train/c001_el",
        "NetMHCpan_train/c002_el",
        "NetMHCpan_train/c003_el",
        "NetMHCpan_train/c004_el",
        "mhcmotifatlas.all_peptides.txt.tsv",
        "1-s2.0-S2405471222004707-mmc4.txt", # MixMHCpred
        "1-s2.0-S2405471222004707-mmc6.xlsx", # PRIME
        "mhcflurry_4_2.2.0/curated_training_data.affinity.csv.bz2",
        "mhcflurry_4_2.2.0/curated_training_data.csv.bz2",
        "mhcflurry_4_2.2.0/curated_training_data.mass_spec.csv.bz2",
        "mhcflurry_4_2.2.0/curated_training_data.no_additional_ms.csv.bz2",
    ]
    
    # 2. 验证文件是否存在
    print("1. 验证文件路径...")
    if not os.path.exists(USER_FILE):
        print(f"错误：用户文件 {USER_FILE} 不存在！")
        print("请检查文件路径是否正确。")
        return
    
    # 过滤存在的训练文件
    existing_training_files = []
    for file in TRAINING_FILES:
        if os.path.exists(file):
            existing_training_files.append(file)
        else:
            print(f"警告：训练文件 {file} 不存在，已跳过")
    
    if not existing_training_files:
        print("错误：没有找到任何训练数据文件！")
        return
    
    print(f"找到 {len(existing_training_files)} 个训练数据文件")
    
    # 3. 提取肽段序列
    print("\n2. 提取用户肽段序列...")
    user_peptides = extract_peptides_from_user_file(USER_FILE)
    
    print("\n3. 提取训练数据肽段序列...")
    training_peptides = extract_peptides_from_training_files(existing_training_files)
    
    if not user_peptides:
        print("错误：未从用户文件中提取到有效肽段！")
        return
    
    if not training_peptides:
        print("错误：未从训练文件中提取到有效肽段！")
        return
    
    # 4. 检测泄露
    print("\n4. 正在检测数据泄露...")
    leakage_result = find_leaked_peptides(user_peptides, training_peptides)
    
    # 5. 生成报告
    print("\n5. 生成检测报告...")
    report_path, csv_path = generate_report(leakage_result)
    
    # 6. 显示摘要
    print("\n" + "=" * 60)
    print("检测完成！摘要如下：")
    print("=" * 60)
    print(f"用户肽段总数: {leakage_result['total_user_peptides']}")
    print(f"训练数据肽段总数: {leakage_result['total_training_peptides']}")
    print(f"可能泄露的肽段数: {leakage_result['num_leaked']}")
    print(f"泄露率: {leakage_result['leakage_rate']:.2f}%")
    print(f"\n详细报告: {report_path}")
    print(f"CSV结果文件: {csv_path}")
    
    # 风险提示
    if leakage_result['leakage_rate'] > 50:
        print("\n⚠️  高风险警告：超过50%的肽段存在于训练数据中，可能存在严重的数据泄露！")
    elif leakage_result['leakage_rate'] > 10:
        print("\n⚠️  中等风险：10%-50%的肽段存在于训练数据中，建议进一步核查！")
    else:
        print("\n✅ 低风险：泄露率低于10%，数据安全性较好！")

if __name__ == "__main__":
    main()
