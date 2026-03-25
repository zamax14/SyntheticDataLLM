# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: merge.py
# Authors: Abraham Sánchez, Ulises Moya, Alejandro Zarate
# Description:
#   CLI para fusionar todos los archivos CSV de un directorio en un único
#   archivo CSV de salida.
# License: MIT
# ==============================================================================

import os

import pandas as pd
from dataclasses import dataclass
from jsonargparse import CLI
from tqdm import tqdm

from utils.logger import Logger
from utils.utils import CSVExtension, save_csv


@dataclass
class MergeCSV:
    """Clase para fusionar múltiples archivos CSV en uno solo."""

    def merge(
        self,
        data_path: str,
        output_path: str,
        output_filename: str = 'merged_qa',
    ) -> None:
        """
        Fusiona todos los archivos CSV de un directorio en un único CSV.

        Args:
            data_path (str): Directorio de entrada que contiene los archivos CSV.
            output_path (str): Directorio de salida donde se guardará el CSV fusionado.
            output_filename (str): Nombre base del archivo de salida (sin extensión).

        Returns:
            None
        """
        archivos = sorted([
            f for f in os.listdir(data_path)
            if f.lower().endswith(CSVExtension)
        ])

        if not archivos:
            Logger.warning(f"No se encontraron archivos CSV en: {data_path}")
            return

        Logger.info(f"Archivos CSV encontrados: {len(archivos)}")

        fragmentos: list[pd.DataFrame] = []

        for archivo in tqdm(archivos, desc="Leyendo CSVs"):
            ruta = os.path.join(data_path, archivo)
            try:
                df = pd.read_csv(ruta)
                fragmentos.append(df)
            except Exception as e:
                Logger.warning(f"No se pudo leer '{archivo}': {e}")

        if not fragmentos:
            Logger.error("No se pudo cargar ningún CSV. Abortando.")
            return

        df_merged = pd.concat(fragmentos, ignore_index=True)
        Logger.info(f"Total de registros fusionados: {len(df_merged)}")

        os.makedirs(output_path, exist_ok=True)
        save_csv(df_merged, output_filename, output_path)
        Logger.info(f"CSV fusionado guardado en: {os.path.join(output_path, output_filename + CSVExtension)}")


if __name__ == '__main__':
    CLI(MergeCSV)
