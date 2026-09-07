import re
import os
import pandas as pd


class FilenameHelper:
    """Helper class for sanitizing filenames to work on all platforms."""

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Remove invalid Windows filename characters.
        Invalid chars: < > : " / \\ | ? *
        Replaces them with underscores.
        """
        invalid_chars = r'[<>:"/\\|?*]'
        sanitized = re.sub(invalid_chars, '_', filename)
        return sanitized.strip(' .')


class UtilityHelper:

    @staticmethod
    def latex_to_html_table(latex_file):
        with open(latex_file) as f:
            content = f.read()

        content = re.sub(r"\\begin{tabular}{.*?}", "", content)
        content = re.sub(r"\\end{tabular}", "", content)
        for cmd in ["\\toprule", "\\midrule", "\\bottomrule"]:
            content = content.replace(cmd, "")

        rows = [r.strip() for r in content.split("\\\\") if r.strip()]
        if rows and rows[0].startswith("{") and rows[0].endswith("}"):
            rows = rows[1:]

        data = [[c.strip() for c in row.split("&")] for row in rows]
        if not data:
            return "<p>Empty table</p>"
        df = pd.DataFrame(data[1:], columns=data[0])
        return df.to_html(index=False, escape=False)

    @staticmethod
    def create_html_report(save_folder="test_results", output_file="report.html"):
        html_file = os.path.join(save_folder, output_file)
        html_content = "<html><head><title>Robustness Report</title></head><body>\n"
        html_content += "<h1>Robustness &amp; Resilience Report</h1>\n"

        for name, filename in [
            ("Robustness",                        "robustness_table.tex"),
            ("Resilience (Reward)",               "reward_table.tex"),
            ("Resilience (Observation Similarity)", "observation_table.tex"),
        ]:
            path = os.path.join(save_folder, filename)
            html_content += f"<h2>{name}</h2>\n"
            if os.path.exists(path):
                html_content += UtilityHelper.latex_to_html_table(path)
            else:
                html_content += f"<p>Missing table: {filename}</p>\n"

        html_content += "<h2>Plots</h2>\n"

        overall_svgs = sorted([f for f in os.listdir(save_folder) if f.endswith(".svg")])
        if overall_svgs:
            html_content += "<h3>Overall Plots</h3>\n"
            for file in overall_svgs:
                html_content += f'<img src="{file}" alt="{file}" style="max-width:800px;"><br>\n'

        episode_dirs = sorted([d for d in os.listdir(save_folder) if d.lower().startswith("episode")])
        for ep_dir in episode_dirs:
            ep_path = os.path.join(save_folder, ep_dir)
            if os.path.isdir(ep_path):
                html_content += f"<h3>{ep_dir}</h3>\n"
                for file in sorted(os.listdir(ep_path)):
                    if file.endswith(".svg"):
                        rel = os.path.relpath(
                            os.path.join(ep_path, file), start=save_folder
                        ).replace("\\", "/").replace(" ", "%20")
                        html_content += f'<img src="{rel}" alt="{file}" style="max-width:800px;"><br>\n'

        html_content += "</body></html>"
        with open(html_file, "w") as f:
            f.write(html_content)
        print(f"[INFO] HTML report saved to {html_file}")
