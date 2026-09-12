from pathlib import Path

def modify_file(input_path: Path) -> None:
    output_path = input_path.with_name(input_path.stem + "_modified.txt")

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.strip()

            # Keep empty lines as they are
            if not stripped:
                fout.write(line)
                continue

            # Keep header line unchanged
            if stripped.startswith("#"):
                fout.write(line)
                continue

            parts = stripped.split()

            # Expecting at least 3 columns: tau_total, W_out, Q_meas
            if len(parts) >= 3:
                try:
                    tau_total = parts[0]
                    W_out = float(parts[1])
                    Q_meas = parts[2]

                    # Flip the sign of the second column
                    W_out_modified = -W_out

                    # Preserve any extra columns if present
                    extra = parts[3:]
                    new_parts = [tau_total, f"{W_out_modified:.18e}", Q_meas] + extra
                    fout.write(" ".join(new_parts) + "\n")
                except ValueError:
                    # If conversion fails, write the line unchanged
                    fout.write(line)
            else:
                fout.write(line)

def main():
    folder = Path(".")  # change this if your files are in another directory

    # Match files like:
    # Otto_lubricated_Fluctuating_Gamma_40.0_measurements_200_trajectories_50_tau_from_5_to_10.txt
    pattern = "Otto_lubricated_Fluctuating_Gamma_*_measurements_*_trajectories_50_tau_from_5_to_10.txt"

    for file_path in folder.glob(pattern):
        modify_file(file_path)
        print(f"Modified: {file_path.name}")

if __name__ == "__main__":
    main()