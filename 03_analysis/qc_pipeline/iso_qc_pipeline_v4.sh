#!/usr/bin/env bash

set -e

echo "=== STEP 1: Generate codon-optimised ttrR and ttrS FASTA files ==="
python3 - << 'EOF'
import re

ecoli_codon_table = {
    'A': 'GCG', 'R': 'CGT', 'N': 'AAC', 'D': 'GAT', 'C': 'TGC',
    'Q': 'CAG', 'E': 'GAA', 'G': 'GGC', 'H': 'CAT', 'I': 'ATT',
    'L': 'CTG', 'K': 'AAA', 'M': 'ATG', 'F': 'TTT', 'P': 'CCG',
    'S': 'AGC', 'T': 'ACC', 'W': 'TGG', 'Y': 'TAT', 'V': 'GTT',
    '*': 'TAA'
}

def codon_optimise(protein_seq):
    protein_seq = protein_seq.strip().upper()
    if protein_seq.endswith('*'):
        protein_seq = protein_seq[:-1]
    dna = ''.join(ecoli_codon_table.get(aa, 'NNN') for aa in protein_seq)
    dna += ecoli_codon_table['*']
    return dna

def write_fasta(filepath, header, dna):
    gc = (dna.count('G') + dna.count('C')) / len(dna)
    print(f"  {filepath}: {len(dna)} bp, GC={round(gc*100,1)}%")
    with open(filepath, 'w') as f:
        f.write(f'>{header}\n')
        for i in range(0, len(dna), 60):
            f.write(dna[i:i+60] + '\n')

import os
if os.path.exists("ttrR_optimized.fasta") and os.path.exists("ttrS_optimized.fasta"):
    print("  ttrR_optimized.fasta and ttrS_optimized.fasta already present — skipping regeneration")
else:
    ttrR_protein = "MATIHLLDDDTAVTNACAFLLESLGYDVKCWTQGADFLAQASLYQAGVVLLDMRMPVLDGQGVHDALRQCGSTLAVVFLTGHGDVPMAVEQMKRGAVDFLQKPVSVKPLQAALERALTVSSAAVARRE IILCYQQLTPKERELASLVAKGFMNREIAEMNIAVRTVE VHRARVMEKMQAGSLAELIR RFEKMASPETRIRTTYEP"
    ttrR_protein = ttrR_protein.replace(' ', '')
    write_fasta("ttrR_optimized.fasta", "ttrR_LT2_EcN_codon_optimised", codon_optimise(ttrR_protein))

    ttrS_protein = "MRGKTVRRLAVLAAVGLLCHGAWAGTWNIGILA MRGEASTRSHWQPLAKTLSQQLPGETFHIQPLDLHQMQEAVNQGTVQFVITNPAQFVQLNSHAPLRWLASLRSTRDGKAVSNVIGSVILTRRDSGI TTAHDLIGKTVGAIDAQAFGGYLLGYKALSDA GLRPERDFHLRFTGFPGDALVYMLREKAVQAAIVPVCLLENMDQEGLINKK DFIALLSRPTPLPCLTSTPLYPD WSFAALPAVSDALADRVTRALFNAPAAA SFHWGAPASTSQVEALLRDVRQHPQQRRLWLDVKSWLIQHQLMVGGVILAFLLLTLNYIWVMLLVRRRGKQLERNSVVLHQHERALETARQMSVLGEMTSGFAHELNQPLSAIRHYAQGCLIRLRAADEQHPLLPALEQIDQQAQRGADTLRNLRHWVSQAQGNPVLTEAWKAIAIREAT DHVWQLLRMAQQ FPTVTLHTEVSA ALRVTLPSVLLEQVLANIILNAAQAGATH LWIVAE RTENGISIVLQDNAGGIDEALLRQAFQPFMTTRKEGMGLGLAICQRLVRYGRGDISIR NQTAPDGLSGTVVTIHFLHENGCRDGDNSSTG"
    ttrS_protein = ttrS_protein.replace(' ', '')
    write_fasta("ttrS_optimized.fasta", "ttrS_LT2_EcN_codon_optimised", codon_optimise(ttrS_protein))
EOF

echo "=== STEP 2: Clean and reformat all source FASTA files ==="
python3 - << 'EOF'
import re, os

fasta_files = [
    "pSC101_KanR_MCS.fasta",
    "BBa_J23115.fasta",
    "BBa_B0031.fasta",
    "ttrR_optimized.fasta",
    "BBa_B0015.fasta",
    "BBa_J23106.fasta",
    "BBa_B0031.fasta",
    "ttrS_optimized.fasta",
    "BBa_B0015.fasta",
    "pTtr_Salmonella_LT2.fasta",
    "BBa_B0031.fasta",
    "BBa_I746916.fasta",
    "BBa_B0015.fasta",
]

seen = set()
for fname in fasta_files:
    if fname in seen or not os.path.exists(fname):
        continue
    seen.add(fname)
    with open(fname) as f:
        raw = f.read()
    entries = re.split(r'(?=>)', raw)
    entries = [e.strip() for e in entries if e.strip()]
    clean_entries = []
    for entry in entries:
        lines = entry.split('\n')
        header = lines[0] if lines[0].startswith('>') else None
        if not header:
            continue
        seq_lines = []
        for line in lines[1:]:
            line = re.sub(r'>.*', '', line.strip())
            cleaned = re.sub(r'[^ATGCatgcNn]', '', line)
            if cleaned:
                seq_lines.append(cleaned.upper())
        seq = ''.join(seq_lines)
        if seq:
            clean_entries.append((header, seq))
    cleaned_fname = fname.replace('.fasta', '_clean.fasta')
    with open(cleaned_fname, 'w') as out:
        for header, seq in clean_entries:
            out.write(header + '\n')
            for i in range(0, len(seq), 60):
                out.write(seq[i:i+60] + '\n')
    total_bp = sum(len(s) for _, s in clean_entries)
    print(f"  {fname}: {len(clean_entries)} entry, {total_bp} bp -> {cleaned_fname}")
print("  Cleaning complete.")
EOF

echo "=== STEP 3: Assemble full construct ==="
cat \
  pSC101_KanR_MCS_clean.fasta \
  BBa_J23115_clean.fasta \
  BBa_B0031_clean.fasta \
  ttrR_optimized_clean.fasta \
  BBa_B0015_clean.fasta \
  BBa_J23106_clean.fasta \
  BBa_B0031_clean.fasta \
  ttrS_optimized_clean.fasta \
  BBa_B0015_clean.fasta \
  pTtr_Salmonella_LT2_clean.fasta \
  BBa_B0031_clean.fasta \
  BBa_I746916_clean.fasta \
  BBa_B0015_clean.fasta \
  > full_construct_ordered.fasta

echo "=== STEP 4: Length sanity check ==="
python3 - << 'EOF'
parts = [
    ("pSC101_KanR_MCS",    4509),
    ("BBa_J23115",           35),
    ("BBa_B0031",            14),
    ("ttrR_optimized",      618),
    ("BBa_B0015",           129),
    ("BBa_J23106",           35),
    ("BBa_B0031",            14),
    ("ttrS_optimized",     1779),
    ("BBa_B0015",           129),
    ("pTtr_Salmonella_LT2", 201),
    ("BBa_B0031",            14),
    ("BBa_I746916",         720),
    ("BBa_B0015",           129),
]
expected = sum(l for _, l in parts)
seq = ""
with open("full_construct_ordered.fasta") as f:
    for line in f:
        line = line.strip()
        if not line.startswith(">"):
            seq += line.upper()
print(f"  Expected: {expected} bp")
print(f"  Actual:   {len(seq)} bp")
diff = abs(len(seq) - expected)
if diff == 0:
    print("  OK: exact match")
elif diff <= 20:
    print(f"  OK: within tolerance ({diff}bp difference)")
else:
    print(f"  WARNING: {diff}bp discrepancy — check Twist file lengths")
EOF

echo "=== STEP 5: GC + homopolymer + repeat analysis ==="
python3 - << 'EOF'
import re
from collections import Counter

seq = ""
with open("full_construct_ordered.fasta") as f:
    for line in f:
        line = line.strip()
        if not line.startswith(">"):
            seq += line.upper()

total = len(seq)
gc = (seq.count("G") + seq.count("C")) / total
print(f"\n[Construct length] {total} bp")
print(f"[GC content] {round(gc*100,1)}%  {'OK' if 0.35 <= gc <= 0.65 else 'WARNING'}")

parts = [
    ("pSC101_KanR_MCS",    4509), ("BBa_J23115",35), ("BBa_B0031",14),
    ("ttrR_optimized",      618), ("BBa_B0015",129), ("BBa_J23106",35),
    ("BBa_B0031",            14), ("ttrS_optimized",1779), ("BBa_B0015",129),
    ("pTtr_Salmonella_LT2", 201), ("BBa_B0031",14), ("BBa_I746916",720),
    ("BBa_B0015",           129),
]
boundaries, pos = [], 0
for name, length in parts:
    boundaries.append((pos, pos+length, name))
    pos += length

def which_part(p):
    for s, e, n in boundaries:
        if s <= p < e: return n
    return "beyond-map"

print("\n[Homopolymer scan]")
hp_all = list(re.finditer(r"([ATGC])\1{5,}", seq))
backbone_end = 4509
if hp_all:
    for m in hp_all:
        base, ln, pos = m.group()[0], len(m.group()), m.start()
        sev = "CRITICAL" if ln>=10 else ("REVIEW" if ln>=8 else "FLAG")
        print(f"  [{sev}] {base}x{ln} pos {pos} — {which_part(pos)}")
else:
    print("  None detected")

kmers = Counter(seq[i:i+8] for i in range(len(seq)-8))
repeats = {k:v for k,v in kmers.items() if v > 3}
print(f"\n[8-mer repeats >3x] {len(repeats)} found")
for k,v in sorted(repeats.items(), key=lambda x:-x[1])[:10]:
    print(f"  {k} -> {v}x")
EOF

echo "=== STEP 6: Part count verification ==="
echo "J23115:  $(grep -oc 'J23115' full_construct_ordered.fasta || echo 0) (expected: 1)"
echo "J23106:  $(grep -oc 'J23106' full_construct_ordered.fasta || echo 0) (expected: 1)"
echo "B0015:   $(grep -oc 'B0015'  full_construct_ordered.fasta || echo 0) (expected: 3)"
echo "B0031:   $(grep -oc 'B0031'  full_construct_ordered.fasta || echo 0) (expected: 3)"

echo "=== STEP 7: Synthesis risk summary ==="
python3 - << 'EOF'
import re
from collections import Counter

seq = ""
with open("full_construct_ordered.fasta") as f:
    for line in f:
        line = line.strip()
        if not line.startswith(">"):
            seq += line.upper()

risk, notes = 0, []
gc = (seq.count("G") + seq.count("C")) / len(seq)
if gc < 0.35 or gc > 0.65:
    risk += 2; notes.append(f"GC={round(gc*100,1)}% outside safe range")
else:
    notes.append(f"GC={round(gc*100,1)}% OK")

backbone_end = 4509
hp_all = list(re.finditer(r"([ATGC])\1{5,}", seq))
hp_critical = [m for m in hp_all if len(m.group()) >= 10]
hp_gc_insert = [m for m in hp_all if m.group()[0] in "GC" and m.start() >= backbone_end]
if hp_critical:
    risk += len(hp_critical)*2; notes.append(f"{len(hp_critical)} homopolymer(s) >=10bp")
if hp_gc_insert:
    risk += min(len(hp_gc_insert),2); notes.append(f"{len(hp_gc_insert)} G/C run(s) in insert")
elif not hp_gc_insert:
    notes.append("No G/C runs in insert region")

kmers = Counter(seq[i:i+8] for i in range(len(seq)-8))
high_rep = sum(1 for v in kmers.values() if v > 5)
if high_rep > 20:
    risk += 1; notes.append(f"{high_rep} high-frequency 8-mers — review")
else:
    notes.append(f"{high_rep} high-frequency 8-mers (acceptable)")

score = min(risk, 5)
print(f"\n[Synthesis risk score] {score}/5")
for n in notes: print(f"  - {n}")
print("\nVerdict: LOW RISK — proceed to Twist order" if score <= 1 else
      "\nVerdict: MODERATE RISK — review flags" if score <= 3 else
      "\nVerdict: HIGH RISK — resolve before ordering")
EOF

echo ""
echo "=== QC complete — proceed to Tellurium ODE model update ==="
