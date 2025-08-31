"""
Add active positions display under each asset card
"""

# Read the dashboard HTML
filepath = 'tradingbot/ui/templates/dashboard.html'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Define the positions display HTML to add after metrics-grid
positions_html = '''
                        <!-- Active Positions -->
                        <div class="positions-section" id="{asset}-{mode}-positions-section" style="margin-top: 10px; display: none;">
                            <div class="positions-header" style="font-size: 12px; color: #8892b0; margin-bottom: 5px;">
                                <i class="fas fa-chart-line"></i> Active Positions
                            </div>
                            <div class="positions-list" id="{asset}-{mode}-positions-list" style="max-height: 100px; overflow-y: auto;">
                                <!-- Positions will be populated here -->
                            </div>
                        </div>'''

# Find all metrics-grid sections and add positions after them
import re

# Pattern to find the end of metrics-grid div
pattern = r'(</div>\s*<!--\s*metrics-grid\s*-->|</div>\s*</div>\s*</div>\s*<!--\s*trading-content)'

# Counter for replacements
replacements = 0

# Function to add positions display
def add_positions(match):
    global replacements
    replacements += 1
    
    # Determine asset and mode based on position in file
    # This is a simple approach - in production you'd parse the HTML properly
    assets = ['crypto', 'futures', 'forex', 'forex_options']
    modes = ['paper', 'live']
    
    asset_idx = (replacements - 1) // 2
    mode_idx = (replacements - 1) % 2
    
    if asset_idx < len(assets):
        asset = assets[asset_idx]
        mode = modes[mode_idx]
        
        positions_with_ids = positions_html.replace('{asset}', asset).replace('{mode}', mode)
        return '</div>' + positions_with_ids
    
    return match.group(0)

# For a more precise approach, let's manually add the positions sections
# Find each trading-content section and add positions

# Pattern to find the end of metrics-grid within trading sections
sections_to_update = [
    ('crypto-paper', r'(<div class="trading-content">.*?<div class="metrics-grid">.*?</div>)', 'paper'),
    ('crypto-live', r'(<!-- Live Trading Section -->.*?<div class="metrics-grid">.*?</div>)', 'live'),
    ('futures-paper', r'(<!-- Crypto Futures Card -->.*?<!-- Paper Trading Section -->.*?<div class="metrics-grid">.*?</div>)', 'paper'),
    ('futures-live', r'(<!-- Live Trading Section -->.*?<!-- Crypto Futures Card -->.*?<div class="metrics-grid">.*?</div>)', 'live'),
]

# Simpler approach - add after each metric-grid closing div
# Count occurrences
metric_grids = list(re.finditer(r'<div class="metrics-grid">.*?</div>', content, re.DOTALL))

# Assets and modes in order they appear
asset_mode_pairs = [
    ('crypto', 'paper'), ('crypto', 'live'),
    ('futures', 'paper'), ('futures', 'live'),
    ('forex', 'paper'), ('forex', 'live'),
    ('forex_options', 'paper'), ('forex_options', 'live')
]

# Replace from end to beginning to maintain positions
for i in range(len(metric_grids)-1, -1, -1):
    if i < len(asset_mode_pairs):
        asset, mode = asset_mode_pairs[i]
        end_pos = metric_grids[i].end()
        
        positions_section = positions_html.replace('{asset}', asset).replace('{mode}', mode)
        content = content[:end_pos] + positions_section + content[end_pos:]

# Write back
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Added active positions display sections to dashboard")
print("Positions will be shown under each asset when there are active trades")