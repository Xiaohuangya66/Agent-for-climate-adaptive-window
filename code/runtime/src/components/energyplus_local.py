"""
EnergyPlus Runner (移植自本地成功实现)

完整的EnergyPlus运行和解析逻辑：
- 真实UDI解析（从eplusout.csv）
- 真实能耗解析（Electricity + NaturalGas）
- 严格的成功判定
- 运行目录保留
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, Optional
import pandas as pd


SUCCESS_MARKER = "EnergyPlus Completed Successfully"
SEVERE_SUMMARY = re.compile(r"(\d+)\s+Severe Errors?", re.IGNORECASE)


class EnergyPlusRunner:
    """EnergyPlus运行器（移植自本地成功实现）"""
    
    def __init__(self, 
                 energyplus_path: str,
                 base_idf_path: Path,
                 epw_dir: Path,
                 output_base_dir: Path):
        """
        Args:
            energyplus_path: EnergyPlus可执行文件路径
            base_idf_path: 基础IDF文件路径
            epw_dir: EPW文件目录
            output_base_dir: 输出基础目录
        """
        self.energyplus_path = energyplus_path
        self.base_idf_path = Path(base_idf_path)
        self.epw_dir = Path(epw_dir)
        self.output_base_dir = Path(output_base_dir)
        
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, 
            structure: str,
            cold_spectrum: Dict,
            hot_spectrum: Dict,
            city: str,
            candidate_id: str = "test") -> Dict:
        """
        运行EnergyPlus仿真
        
        Args:
            structure: 结构字符串
            cold_spectrum: 冷态光谱 {'wavelengths_nm', 'T', 'Rf', 'Rb'}
            hot_spectrum: 热态光谱
            city: 城市名称
            candidate_id: 候选ID
            
        Returns:
            结果字典，包含UDI、能耗等
        """
        # 创建运行目录
        run_dir = self.output_base_dir / f"{city}_{candidate_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成修改后的IDF
        modified_idf = run_dir / "modified.idf"
        self._write_idf_with_spectra(
            modified_idf, 
            cold_spectrum, 
            hot_spectrum
        )
        
        # EPW文件
        epw_file = self.epw_dir / f"{city}.epw"
        if not epw_file.exists():
            return {
                'status': 'failed',
                'error': f'EPW file not found: {epw_file}',
                'UDI': None,
                'TotalEnergy': None
            }
        
        # 运行EnergyPlus
        try:
            result = subprocess.run(
                [
                    self.energyplus_path,
                    '-w', str(epw_file),
                    '-d', str(run_dir),
                    '-r',
                    str(modified_idf)
                ],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # 保存输出
            (run_dir / "stdout.log").write_text(result.stdout)
            (run_dir / "stderr.log").write_text(result.stderr)
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'failed',
                'error': 'EnergyPlus timeout',
                'UDI': None,
                'TotalEnergy': None
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': f'EnergyPlus execution error: {e}',
                'UDI': None,
                'TotalEnergy': None
            }
        
        # 解析结果
        return self._parse_results(run_dir, result.returncode)
    
    def _write_idf_with_spectra(self, 
                                 output_path: Path,
                                 cold_spectrum: Dict,
                                 hot_spectrum: Dict):
        """写入带光谱数据的IDF文件"""
        # 读取基础IDF
        base_content = self.base_idf_path.read_text()
        
        # 生成光谱块
        cold_spec = self._format_spectrum_block("VO2_Cold_Spec", cold_spectrum)
        hot_spec = self._format_spectrum_block("VO2_Hot_Spec", hot_spectrum)
        
        # 添加输出请求
        output_requests = """
Output:Variable,
    *,
    Daylighting Reference Point 1 Illuminance,
    Hourly;

Output:Variable,
    *,
    Site Solar Altitude Angle,
    Hourly;

Output:Meter,
    Electricity:Facility,
    RunPeriod;

Output:Meter,
    NaturalGas:Facility,
    RunPeriod;
"""
        
        # 组合
        full_content = base_content + "\n" + cold_spec + "\n" + hot_spec + "\n" + output_requests
        
        output_path.write_text(full_content)
    
    def _format_spectrum_block(self, name: str, spectrum: Dict) -> str:
        """格式化光谱数据块"""
        lines = [f"MaterialProperty:GlazingSpectralData,"]
        lines.append(f"    {name},")
        
        wavelengths_nm = spectrum['wavelengths_nm']
        T = spectrum['T']
        Rf = spectrum['Rf']
        Rb = spectrum['Rb']
        
        for i, wl_nm in enumerate(wavelengths_nm):
            wl_um = wl_nm / 1000.0
            t = T[i]
            rf = Rf[i]
            rb = Rb[i]
            
            line = f"    {wl_um:.4f}, {t:.6f}, {rf:.6f}, {rb:.6f}"
            if i < len(wavelengths_nm) - 1:
                line += ","
            else:
                line += ";"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _parse_results(self, run_dir: Path, return_code: int) -> Dict:
        """解析EnergyPlus输出"""
        end_file = run_dir / "eplusout.end"
        err_file = run_dir / "eplusout.err"
        csv_file = run_dir / "eplusout.csv"
        
        # 读取状态文件
        end_text = end_file.read_text() if end_file.exists() else ""
        err_text = err_file.read_text() if err_file.exists() else ""
        
        # 检查成功标记
        success_marker = SUCCESS_MARKER.lower() in end_text.lower()
        fatal_detected = "**  Fatal  **" in err_text or "Fatal Error" in err_text
        
        # 检查Severe错误
        severe_matches = SEVERE_SUMMARY.findall(end_text + "\n" + err_text)
        severe_errors = int(severe_matches[-1]) if severe_matches else 0
        
        # 判定失败条件
        if return_code != 0:
            return {
                'status': 'failed',
                'error': f'Non-zero return code: {return_code}',
                'UDI': None,
                'TotalEnergy': None
            }
        
        if not success_marker or fatal_detected or severe_errors > 0:
            return {
                'status': 'failed',
                'error': f'EnergyPlus not successful (severe={severe_errors}, fatal={fatal_detected})',
                'UDI': None,
                'TotalEnergy': None
            }
        
        if not csv_file.exists():
            return {
                'status': 'failed',
                'error': 'eplusout.csv not found',
                'UDI': None,
                'TotalEnergy': None
            }
        
        # 解析CSV
        try:
            udi = self._parse_udi(csv_file)
            total_energy = self._parse_energy(csv_file)
            
            if udi is None or total_energy is None:
                return {
                    'status': 'failed',
                    'error': 'Failed to parse UDI or energy',
                    'UDI': udi,
                    'TotalEnergy': total_energy
                }
            
            return {
                'status': 'success',
                'UDI': float(udi),
                'TotalEnergy': float(total_energy),
                'run_dir': str(run_dir)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': f'Parsing error: {e}',
                'UDI': None,
                'TotalEnergy': None
            }
    
    def _parse_udi(self, csv_file: Path) -> Optional[float]:
        """从CSV解析真实UDI"""
        try:
            df = pd.read_csv(csv_file)
            
            # 查找照度列
            illum_cols = [col for col in df.columns if 'Illuminance' in col or 'illuminance' in col]
            
            if not illum_cols:
                return None
            
            # 使用第一个照度列
            illum_col = illum_cols[0]
            illuminance = df[illum_col].dropna()
            
            if len(illuminance) == 0:
                return None
            
            # 计算UDI (100-2000 lux)
            in_range = ((illuminance >= 100) & (illuminance <= 2000)).sum()
            udi = in_range / len(illuminance)
            
            return float(udi)
            
        except Exception as e:
            print(f"Warning: UDI parsing failed: {e}")
            return None
    
    def _parse_energy(self, csv_file: Path) -> Optional[float]:
        """从CSV解析真实能耗"""
        try:
            df = pd.read_csv(csv_file)
            
            # 查找Electricity:Facility
            elec_cols = [col for col in df.columns if 'Electricity:Facility' in col and 'RunPeriod' in col]
            gas_cols = [col for col in df.columns if 'NaturalGas:Facility' in col and 'RunPeriod' in col]
            
            electricity = 0.0
            natural_gas = 0.0
            
            if elec_cols:
                elec_col = elec_cols[0]
                elec_values = df[elec_col].dropna()
                if len(elec_values) > 0:
                    electricity = float(elec_values.iloc[-1])
            
            if gas_cols:
                gas_col = gas_cols[0]
                gas_values = df[gas_col].dropna()
                if len(gas_values) > 0:
                    natural_gas = float(gas_values.iloc[-1])
            
            # 总能耗 = 电 + 气
            total_energy = electricity + natural_gas
            
            if total_energy == 0:
                return None
            
            return float(total_energy)
            
        except Exception as e:
            print(f"Warning: Energy parsing failed: {e}")
            return None
