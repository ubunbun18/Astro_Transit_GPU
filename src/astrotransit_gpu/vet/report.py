import os
import json
import pandas as pd
from datetime import datetime

def save_summary_json(df, out_dir, meta):
    """
    Saves a machine-readable summary of the vetting run.
    """
    snr_thresh = float(meta.get('snr_threshold', 7.1))
    
    summary = {
        "generated_at": datetime.now().isoformat(),
        "input_results": meta.get('input_results'),
        "config": meta.get('config_path'),
        "snr_threshold": snr_thresh,
        "total_candidates": len(df),
        "significant_signals": len(df[df['snr'] >= snr_thresh]),
        "known_tois": len(df[df['known_type'] == 'toi']),
        "known_ebs": len(df[df['known_type'] == 'eb']),
        "unknown_candidates": len(df[df['known_type'] == 'unknown']),
        "high_score_unknowns": len(df[(df['known_type'] == 'unknown') & (df['vetting_score'] > 0.7)])
    }
    
    path = os.path.join(out_dir, "summary.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    return path

def generate_html_report(df, out_dir, meta, report_name="index.html"):
    """
    Generates a high-density triage dashboard for candidate vetting.
    """
    # 1. Prepare Notes
    def get_note(row):
        notes = []
        if row['known_type'] == 'unknown' and row['vetting_score'] > 0.8:
            notes.append("High Priority")
        if row['depth'] > 0.05:
            notes.append("Deep (EB?)")
        if row.get('is_harmonic', False):
            notes.append("Harmonic")
        if row['duration'] < 0.05:
            notes.append("Short Transit")
        return ", ".join(notes) if notes else "Plausible"

    df = df.copy()
    df['notes'] = df.apply(get_note, axis=1)

    # 2. Prepare data for the template
    records = df.to_dict(orient='records')
    json_data = json.dumps(records)
    
    # Stats (using normalized SNR)
    snr_thresh = float(meta.get('snr_threshold', 7.1))
    total = len(df)
    tois = len(df[df['known_type'] == 'toi'])
    ebs = len(df[df['known_type'] == 'eb'])
    unknowns = len(df[df['known_type'] == 'unknown'])
    significant = len(df[df['snr'] >= snr_thresh])
    high_score = len(df[(df['known_type'] == 'unknown') & (df['vetting_score'] > 0.7)])

    html_template = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>V39 Candidate Triage Dashboard</title>
    <style>
        :root {{
            --bg-color: #0b0e14;
            --panel-bg: #151921;
            --header-bg: #1a1f29;
            --text-main: #e2e8f0;
            --text-dim: #94a3b8;
            --accent: #3b82f6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --border: #2d3748;
            --row-hover: #1e2530;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, system-ui, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 0;
            font-size: 13px;
            line-height: 1.4;
        }}
        
        header {{
            background: var(--header-bg);
            padding: 15px 25px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        h1 {{ font-size: 18px; margin: 0; font-weight: 700; letter-spacing: -0.02em; }}
        .meta-info {{ font-size: 11px; color: var(--text-dim); text-align: right; }}

        .container {{ padding: 20px 25px; }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}
        
        .stat-card {{
            background: var(--panel-bg);
            padding: 15px;
            border-radius: 6px;
            border: 1px solid var(--border);
        }}
        
        .stat-label {{ font-size: 11px; color: var(--text-dim); text-transform: uppercase; margin-bottom: 5px; }}
        .stat-value {{ font-size: 20px; font-weight: 700; color: var(--accent); }}
        .stat-value.success {{ color: var(--success); }}
        .stat-value.danger {{ color: var(--danger); }}

        .filter-bar {{
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            align-items: center;
        }}
        
        input[type="text"] {{
            background: #1e2530;
            border: 1px solid var(--border);
            color: white;
            padding: 6px 12px;
            border-radius: 4px;
            width: 200px;
            outline: none;
        }}
        
        input[type="text"]:focus {{ border-color: var(--accent); }}

        .btn-filter {{
            background: #2d3748;
            color: var(--text-dim);
            border: 1px solid var(--border);
            padding: 5px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }}
        .btn-filter.active {{ background: var(--accent); color: white; border-color: var(--accent); }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--panel-bg);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}
        
        th {{
            background: #1a1f29;
            color: var(--text-dim);
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            padding: 12px;
            text-align: left;
            cursor: pointer;
            border-bottom: 2px solid var(--border);
        }}
        
        td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); }}
        tr:hover {{ background: var(--row-hover); }}
        
        .badge {{
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        
        .badge-toi {{ background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }}
        .badge-eb {{ background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }}
        .badge-unknown {{ background: rgba(59, 130, 246, 0.2); color: #3b82f6; border: 1px solid #3b82f6; }}
        .badge-harmonic {{ background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; }}

        .score-val {{ font-weight: 700; }}
        .score-high {{ color: var(--success); }}

        .note-text {{ font-size: 11px; color: var(--text-dim); }}

        .btn-view {{
            background: transparent;
            color: var(--accent);
            border: 1px solid var(--accent);
            padding: 3px 8px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.2s;
        }}
        .btn-view:hover {{ background: var(--accent); color: white; }}
        .btn-disabled {{ color: #4a5568; border-color: #4a5568; cursor: not-allowed; }}

        /* Modal */
        .modal-overlay {{
            display: none;
            position: fixed;
            top:0; left:0; width:100%; height:100%;
            background: rgba(0,0,0,0.85);
            z-index: 999;
            backdrop-filter: blur(4px);
        }}
        
        .modal-content {{
            display: none;
            position: fixed;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            background: #1a1f29;
            border-radius: 8px;
            border: 1px solid var(--border);
            z-index: 1000;
            width: 850px;
            max-width: 95%;
        }}

        .modal-header {{
            padding: 15px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .modal-body {{ padding: 20px; text-align: center; }}
        .modal-body img {{ max-width: 100%; border-radius: 4px; }}

        .close-btn {{ background: none; border: none; color: var(--text-dim); cursor: pointer; font-size: 20px; }}
    </style>
</head>
<body>
    <header>
        <div>
            <h1>V39 Triage Dashboard</h1>
            <div style="font-size: 11px; color: var(--text-dim)">AstroTransit-GPU Intelligence Engine</div>
        </div>
        <div class="meta-info">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}<br>
            Kernel: {meta.get('kernel_version', 'V39 Apex Predator')}<br>
            Config: {os.path.basename(str(meta.get('config_path', 'default')))}
        </div>
    </header>
    
    <div class="container">
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Signals</div>
                <div class="stat-value">{total}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Significant (> {snr_thresh})</div>
                <div class="stat-value">{significant}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">High Score Unknowns</div>
                <div class="stat-value success">{high_score}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Known TOIs</div>
                <div class="stat-value success">{tois}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Known EBs</div>
                <div class="stat-value danger">{ebs}</div>
            </div>
        </div>

        <div class="filter-bar">
            <input type="text" id="search-input" placeholder="Search TIC ID..." onkeyup="filterTable()">
            <button class="btn-filter" id="btn-unknown" onclick="toggleFilter('unknown')">Unknown Only</button>
            <button class="btn-filter" id="btn-high" onclick="toggleFilter('high')">Score > 0.7</button>
            <button class="btn-filter" onclick="resetFilters()">Reset</button>
        </div>
        
        <table id="candidate-table">
            <thead>
                <tr>
                    <th onclick="sortTable('rank')">Rank</th>
                    <th onclick="sortTable('tic_id')">TIC ID</th>
                    <th onclick="sortTable('known_type')">Type</th>
                    <th onclick="sortTable('period')">Period (d)</th>
                    <th onclick="sortTable('snr')">SNR</th>
                    <th onclick="sortTable('depth')">Depth</th>
                    <th onclick="sortTable('duration')">Duration</th>
                    <th onclick="sortTable('vetting_score')">Score</th>
                    <th>Notes</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody id="table-body"></tbody>
        </table>
    </div>
    
    <div id="modal-overlay" class="modal-overlay" onclick="closeModal()"></div>
    <div id="plot-modal" class="modal-content">
        <div class="modal-header">
            <h3 id="modal-title" style="margin:0; font-size:16px;">Candidate Details</h3>
            <button class="close-btn" onclick="closeModal()">&times;</button>
        </div>
        <div class="modal-body">
            <img id="modal-img" src="" alt="Folded LC">
        </div>
    </div>

    <script>
        const rawData = {json_data};
        // Add rank
        const data = rawData.map((d, i) => ({{ ...d, rank: i + 1 }}));
        
        let currentSort = {{ key: 'rank', desc: false }};
        let activeFilters = {{ unknown: false, high: false }};
        
        function renderTable(list) {{
            const tbody = document.getElementById('table-body');
            tbody.innerHTML = '';
            
            list.forEach(c => {{
                const typeClass = `badge badge-${{c.known_type.toLowerCase()}}`;
                const scoreClass = c.vetting_score > 0.7 ? 'score-high' : '';
                const harmonicTag = c.is_harmonic ? '<span class="badge badge-harmonic">H</span> ' : '';
                
                // Plot handling
                let plotBtn = '';
                if (c.plot_path) {{
                    plotBtn = `<button class="btn-view" onclick="openModal('${{c.plot_path}}', '${{c.tic_id}}', '${{c.period}}')">View Plot</button>`;
                }} else {{
                    plotBtn = `<button class="btn-view btn-disabled" disabled>No Plot</button>`;
                }}
                
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="color:var(--text-dim)">#${{c.rank}}</td>
                    <td style="font-weight:700">${{c.tic_id}}</td>
                    <td><span class="${{typeClass}}">${{c.known_type}}</span></td>
                    <td>${{c.period.toFixed(5)}}</td>
                    <td>${{c.snr.toFixed(2)}}</td>
                    <td>${{(c.depth*100).toFixed(2)}}%</td>
                    <td>${{c.duration.toFixed(4)}}</td>
                    <td class="score-val ${{scoreClass}}">${{c.vetting_score.toFixed(3)}}</td>
                    <td class="note-text">${{harmonicTag}}${{c.notes}}</td>
                    <td>${{plotBtn}}</td>
                `;
                tbody.appendChild(tr);
            }});
        }}
        
        function sortTable(key) {{
            if (currentSort.key === key) currentSort.desc = !currentSort.desc;
            else {{ currentSort.key = key; currentSort.desc = false; }}
            
            applyAll();
        }}
        
        function toggleFilter(type) {{
            activeFilters[type] = !activeFilters[type];
            document.getElementById('btn-' + type).classList.toggle('active', activeFilters[type]);
            applyAll();
        }}
        
        function resetFilters() {{
            activeFilters = {{ unknown: false, high: false }};
            document.getElementById('search-input').value = '';
            document.querySelectorAll('.btn-filter').forEach(b => b.classList.remove('active'));
            applyAll();
        }}
        
        function applyAll() {{
            const search = document.getElementById('search-input').value.toLowerCase();
            let filtered = data.filter(c => {{
                if (search && !c.tic_id.toString().includes(search)) return false;
                if (activeFilters.unknown && c.known_type !== 'unknown') return false;
                if (activeFilters.high && c.vetting_score <= 0.7) return false;
                return true;
            }});
            
            filtered.sort((a, b) => {{
                let valA = a[currentSort.key];
                let valB = b[currentSort.key];
                if (typeof valA === 'string') return currentSort.desc ? valB.localeCompare(valA) : valA.localeCompare(valB);
                return currentSort.desc ? valB - valA : valA - valB;
            }});
            
            renderTable(filtered);
        }}
        
        function openModal(plotPath, ticId, period) {{
            const img = document.getElementById('modal-img');
            img.style.display = 'block';
            
            // Remove any existing placeholder message
            const oldMsg = document.getElementById('placeholder-msg');
            if (oldMsg) oldMsg.remove();

            img.src = plotPath;
            img.onerror = function() {{
                this.style.display = 'none';
                const msg = document.createElement('div');
                msg.id = 'placeholder-msg';
                msg.style.padding = '80px 20px';
                msg.style.background = '#11151c';
                msg.style.color = '#94a3b8';
                msg.style.border = '1px dashed var(--border)';
                msg.style.borderRadius = '4px';
                msg.innerHTML = `
                    <div style="font-size: 24px; margin-bottom: 10px;">📉</div>
                    <div>Plot Not Generated</div>
                    <div style="font-size: 11px; margin-top: 5px;">File not found: ${{plotPath}}</div>
                `;
                this.parentNode.insertBefore(msg, this);
                this.onerror = null;
            }};
            document.getElementById('modal-title').innerText = `TIC ${{ticId}} | P = ${{parseFloat(period).toFixed(5)}} d`;
            document.getElementById('modal-overlay').style.display = 'block';
            document.getElementById('plot-modal').style.display = 'block';
        }}
        
        function closeModal() {{
            document.getElementById('modal-overlay').style.display = 'none';
            document.getElementById('plot-modal').style.display = 'none';
        }}
        
        // Initial load
        applyAll();
    </script>
</body>
</html>"""

    report_path = os.path.join(out_dir, report_name)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    return report_path
