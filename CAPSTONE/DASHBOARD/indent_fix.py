import codecs

with codecs.open('dashboard.py', 'r', 'utf-8') as f:
    lines = f.readlines()

new_lines = []
in_overview = False
in_pred = False

for i, line in enumerate(lines):
    if line.startswith('# ── Page heading ──'):
        in_overview = True

    if line.startswith('# ══════════════════════════════════════════════════════════════════════════════'):
        if in_overview:
            in_overview = False
            in_pred = True
            new_lines.append('elif page == "🤖 Prediksi Nilai":\n')
            
    if in_overview or in_pred:
        # Avoid double-indenting if it's already indented incorrectly by my previous manual attempts
        # Wait, I didn't indent them before. They are currently at 0 spaces!
        # Except the lines I already manually fixed in the sidebar.
        if line.strip():
            new_lines.append('    ' + line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with codecs.open('dashboard.py', 'w', 'utf-8') as f:
    f.writelines(new_lines)
