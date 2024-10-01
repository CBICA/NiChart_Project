import os, pathlib

def main():
    entrypoint_path = pathlib.Path(__file__).parent
    prev_cwd = os.getcwd()
    os.chdir(str(entrypoint_path.absolute()))

    cmd = "streamlit run NiChartProject.py"
    os.system(cmd)
    os.chdir(prev_cwd)

if __name__ == "__main__":
    main()