import re

"""Clean the tsv file to assure each line is accurate"""
def clean_tsv(input_path, output_path):

    cleaned_lines = []
    current_line = ""

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")

            # if line starts with digits + tab, it's a new row
            if re.match(r"^\d+\t", line):
                if current_line:
                    cleaned_lines.append(current_line)
                current_line = line
            else:
                # continuation of previous row — append with space
                current_line += " " + line.strip()

    # append last row
    if current_line:
        cleaned_lines.append(current_line)

    # write cleaned file
    with open(output_path, "w", encoding="utf-8") as f:
        for row in cleaned_lines:
            f.write(row + "\n")