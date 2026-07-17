# ================================================================
# Week 6 Project: Interactive Sales Dashboard
# Topic: Data Visualization Mastery with Seaborn
# Charts: 7 types — Seaborn (5) + Plotly interactive (2)
# ================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, os
warnings.filterwarnings('ignore')

VIZ = 'visualizations'
os.makedirs(VIZ, exist_ok=True)

# ── Global theme ──────────────────────────────────────────────
BG, FG = '#0D1117', 'white'
PALETTE = ['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0']
sns.set_theme(style='darkgrid', palette=PALETTE)

print("=" * 65)
print("   WEEK 6 — INTERACTIVE SALES DASHBOARD (Seaborn + Plotly)")
print("=" * 65)

# ================================================================
# LOAD & PREPARE DATA
# ================================================================
df = pd.read_csv('sales_data.csv')
df['Date']      = pd.to_datetime(df['Date'])
df['Month']     = df['Date'].dt.strftime('%b')
df['MonthNum']  = df['Date'].dt.month
df['Quarter']   = 'Q' + df['Date'].dt.quarter.astype(str)
df['Week']      = df['Date'].dt.isocalendar().week.astype(int)

print(f"\nDataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# Aggregations
month_order   = ['Jan','Feb','Mar','Apr','May','Jun',
                 'Jul','Aug','Sep','Oct','Nov','Dec']
monthly       = (df.groupby(['MonthNum','Month'])['Total_Sales']
                 .sum().reset_index().sort_values('MonthNum'))
product_rev   = df.groupby('Product')['Total_Sales'].sum().sort_values(ascending=False)
region_rev    = df.groupby('Region')['Total_Sales'].sum().sort_values(ascending=False)
prod_reg      = df.pivot_table(values='Total_Sales', index='Region',
                               columns='Product', aggfunc='sum', fill_value=0)
quarter_prod  = df.groupby(['Quarter','Product'])['Total_Sales'].sum().reset_index()

# ================================================================
# DAY 1 — SEABORN: Bar Plot (Sales by Product)
# ================================================================
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor('#161B22')
bar_data = product_rev.reset_index()
bars = sns.barplot(data=bar_data, x='Product', y='Total_Sales',
                   palette=PALETTE, ax=ax, order=product_rev.index)
for p in ax.patches:
    ax.annotate(f"Rs {p.get_height()/1e6:.2f}M",
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='bottom', color=FG, fontsize=9, fontweight='bold',
                xytext=(0, 4), textcoords='offset points')
ax.set_title('Sales Revenue by Product (Seaborn Bar)', color=FG, fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Product', color=FG); ax.set_ylabel('Total Sales (Rs)', color=FG)
ax.tick_params(colors=FG); ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'Rs {x/1e6:.1f}M'))
for sp in ax.spines.values(): sp.set_edgecolor('#30363D')
ax.grid(axis='y', color='#30363D', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{VIZ}/chart1_seaborn_bar.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("\n[Day 1] Chart 1 saved -> chart1_seaborn_bar.png")

# ================================================================
# DAY 2 — SEABORN: Box Plot (Price Distribution by Product)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor('#161B22')
sns.boxplot(data=df, x='Product', y='Price', palette=PALETTE, ax=ax,
            linewidth=1.5, flierprops=dict(marker='o', color='#FF9800', markersize=5))
ax.set_title('Price Distribution by Product (Seaborn Box Plot)', color=FG, fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Product', color=FG); ax.set_ylabel('Price (Rs)', color=FG)
ax.tick_params(colors=FG)
for sp in ax.spines.values(): sp.set_edgecolor('#30363D')
ax.grid(axis='y', color='#30363D', alpha=0.5)
ax.set_facecolor('#161B22')
plt.tight_layout()
plt.savefig(f'{VIZ}/chart2_seaborn_boxplot.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("[Day 2] Chart 2 saved -> chart2_seaborn_boxplot.png")

# ================================================================
# DAY 2 — SEABORN: Violin Plot (Quantity by Region)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor('#161B22')
sns.violinplot(data=df, x='Region', y='Quantity', palette=PALETTE[:4], ax=ax,
               inner='box', linewidth=1.2)
ax.set_title('Quantity Distribution by Region (Seaborn Violin)', color=FG, fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Region', color=FG); ax.set_ylabel('Quantity Sold', color=FG)
ax.tick_params(colors=FG)
for sp in ax.spines.values(): sp.set_edgecolor('#30363D')
ax.set_facecolor('#161B22')
plt.tight_layout()
plt.savefig(f'{VIZ}/chart3_seaborn_violin.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("[Day 2] Chart 3 saved -> chart3_seaborn_violin.png")

# ================================================================
# DAY 3 — SEABORN: Heatmap (Region x Product Pivot)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor('#161B22')
sns.heatmap(prod_reg / 1e6, annot=True, fmt='.2f', cmap='YlOrRd',
            linewidths=0.5, linecolor='#0D1117', ax=ax,
            annot_kws={'size': 10, 'color': 'black'},
            cbar_kws={'label': 'Revenue (Millions Rs)'})
ax.set_title('Correlation Heatmap: Revenue by Region x Product (Rs M)', 
             color=FG, fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Product', color=FG); ax.set_ylabel('Region', color=FG)
ax.tick_params(colors=FG, rotation=0)
plt.tight_layout()
plt.savefig(f'{VIZ}/chart4_seaborn_heatmap.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("[Day 3] Chart 4 saved -> chart4_seaborn_heatmap.png")

# ================================================================
# DAY 4 — SEABORN: Multi-plot Dashboard (2x2 Subplot Grid)
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor(BG)
fig.suptitle('Sales Analytics Dashboard (Multi-plot)', color=FG, fontsize=15, fontweight='bold', y=1.01)

# Subplot 1: Line plot monthly trend
ax = axes[0, 0]; ax.set_facecolor('#161B22')
sns.lineplot(data=monthly, x='Month', y='Total_Sales', ax=ax,
             color='#2196F3', linewidth=2.5, marker='o', markersize=8,
             markerfacecolor='#FF9800')
ax.fill_between(range(len(monthly)), monthly['Total_Sales'], alpha=0.15, color='#2196F3')
ax.set_xticks(range(len(monthly))); ax.set_xticklabels(monthly['Month'], rotation=30, ha='right')
ax.set_title('Monthly Revenue Trend', color=FG, fontsize=11, fontweight='bold')
ax.set_xlabel('Month', color=FG); ax.set_ylabel('Revenue (Rs)', color=FG)
ax.tick_params(colors=FG); ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'Rs {x/1e6:.1f}M'))
for sp in ax.spines.values(): sp.set_edgecolor('#30363D')

# Subplot 2: Count plot by region
ax = axes[0, 1]; ax.set_facecolor('#161B22')
sns.countplot(data=df, x='Region', palette=PALETTE[:4], ax=ax, order=df['Region'].value_counts().index)
ax.set_title('Order Count by Region', color=FG, fontsize=11, fontweight='bold')
ax.set_xlabel('Region', color=FG); ax.set_ylabel('Orders', color=FG)
ax.tick_params(colors=FG)
for sp in ax.spines.values(): sp.set_edgecolor('#30363D')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x()+p.get_width()/2, p.get_height()),
                ha='center', va='bottom', color=FG, fontsize=10)

# Subplot 3: Scatter plot Price vs Total_Sales
ax = axes[1, 0]; ax.set_facecolor('#161B22')
product_colors = {p: PALETTE[i] for i, p in enumerate(df['Product'].unique())}
for prod, grp in df.groupby('Product'):
    ax.scatter(grp['Price'], grp['Total_Sales'], label=prod, alpha=0.7,
               color=product_colors[prod], s=60, edgecolors='white', linewidth=0.4)
ax.set_title('Price vs Total Sales (by Product)', color=FG, fontsize=11, fontweight='bold')
ax.set_xlabel('Price (Rs)', color=FG); ax.set_ylabel('Total Sales (Rs)', color=FG)
ax.tick_params(colors=FG)
ax.legend(facecolor='#161B22', labelcolor=FG, fontsize=8)
for sp in ax.spines.values(): sp.set_edgecolor('#30363D')

# Subplot 4: Horizontal bar — Region revenue
ax = axes[1, 1]; ax.set_facecolor('#161B22')
reg_df = region_rev.reset_index()
sns.barplot(data=reg_df, y='Region', x='Total_Sales', palette=PALETTE[:4], ax=ax, orient='h')
ax.set_title('Revenue by Region', color=FG, fontsize=11, fontweight='bold')
ax.set_xlabel('Revenue (Rs)', color=FG); ax.set_ylabel('Region', color=FG)
ax.tick_params(colors=FG)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'Rs {x/1e6:.1f}M'))
for sp in ax.spines.values(): sp.set_edgecolor('#30363D')

for ax_row in axes:
    for ax in ax_row:
        ax.set_facecolor('#161B22')
        ax.grid(color='#30363D', alpha=0.4)

plt.tight_layout()
plt.savefig(f'{VIZ}/chart5_seaborn_multiplot_dashboard.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("[Day 4] Chart 5 saved -> chart5_seaborn_multiplot_dashboard.png")

# ================================================================
# DAY 5 — PLOTLY: Interactive Bar Chart (hover info)
# ================================================================
fig_px = px.bar(
    product_rev.reset_index(), x='Product', y='Total_Sales',
    color='Product', color_discrete_sequence=PALETTE,
    title='Interactive Product Revenue Dashboard',
    labels={'Total_Sales': 'Revenue (Rs)', 'Product': 'Product Category'},
    text=product_rev.values
)
fig_px.update_traces(
    texttemplate='Rs %{text:,.0f}',
    textposition='outside',
    hovertemplate='<b>%{x}</b><br>Revenue: Rs %{y:,.0f}<extra></extra>'
)
fig_px.update_layout(
    plot_bgcolor='#161B22', paper_bgcolor=BG,
    font=dict(color=FG, size=12),
    title_font=dict(size=15, color=FG),
    showlegend=False,
    xaxis=dict(gridcolor='#30363D', color=FG),
    yaxis=dict(gridcolor='#30363D', color=FG,
               tickformat='Rs ,.0f'),
    hoverlabel=dict(bgcolor='#161B22', font_color=FG)
)
fig_px.write_html(f'{VIZ}/chart6_plotly_interactive_bar.html')
print("[Day 5] Chart 6 saved -> chart6_plotly_interactive_bar.html  (INTERACTIVE)")

# ================================================================
# DAY 6 — PLOTLY: Interactive Multi-chart Dashboard
# ================================================================
fig_dash = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Monthly Revenue Trend', 'Region Share (Pie)',
                    'Product vs Region (Grouped Bar)', 'Quantity vs Price (Scatter)'),
    specs=[[{'type':'scatter'}, {'type':'pie'}],
           [{'type':'bar'},     {'type':'scatter'}]]
)

# Sub 1: Line
fig_dash.add_trace(
    go.Scatter(x=monthly['Month'], y=monthly['Total_Sales'],
               mode='lines+markers', name='Monthly Revenue',
               line=dict(color='#2196F3', width=3),
               marker=dict(size=8, color='#FF9800', symbol='circle'),
               hovertemplate='<b>%{x}</b><br>Revenue: Rs %{y:,.0f}<extra></extra>',
               fill='tozeroy', fillcolor='rgba(33,150,243,0.15)'),
    row=1, col=1
)

# Sub 2: Pie
fig_dash.add_trace(
    go.Pie(labels=region_rev.index.tolist(), values=region_rev.values.tolist(),
           hole=0.35, marker=dict(colors=PALETTE[:4]),
           hovertemplate='<b>%{label}</b><br>Revenue: Rs %{value:,.0f}<br>Share: %{percent}<extra></extra>',
           textinfo='label+percent', name='Region'),
    row=1, col=2
)

# Sub 3: Grouped bar (Product x Region)
for i, product in enumerate(prod_reg.columns):
    fig_dash.add_trace(
        go.Bar(name=product, x=prod_reg.index.tolist(),
               y=(prod_reg[product]/1e6).tolist(),
               marker_color=PALETTE[i % len(PALETTE)],
               hovertemplate=f'<b>{product}</b><br>Region: %{{x}}<br>Rs %{{y:.2f}}M<extra></extra>'),
        row=2, col=1
    )

# Sub 4: Scatter Price vs Quantity
for i, prod in enumerate(df['Product'].unique()):
    subset = df[df['Product'] == prod]
    fig_dash.add_trace(
        go.Scatter(x=subset['Price'], y=subset['Quantity'],
                   mode='markers', name=prod,
                   marker=dict(color=PALETTE[i % len(PALETTE)], size=9, opacity=0.8),
                   hovertemplate=f'<b>{prod}</b><br>Price: Rs %{{x:,.0f}}<br>Qty: %{{y}}<extra></extra>'),
        row=2, col=2
    )

fig_dash.update_layout(
    height=750,
    title=dict(text='Interactive Sales Dashboard — Week 6 Project', font=dict(size=16, color=FG)),
    plot_bgcolor='#161B22', paper_bgcolor=BG,
    font=dict(color=FG, size=11),
    legend=dict(bgcolor='#161B22', font=dict(color=FG), x=1.01),
    hoverlabel=dict(bgcolor='#161B22', font_color=FG),
    barmode='group'
)
for ann in fig_dash.layout.annotations:
    ann.font.color = FG

fig_dash.write_html(f'{VIZ}/chart7_plotly_full_dashboard.html')
print("[Day 6] Chart 7 saved -> chart7_plotly_full_dashboard.html  (INTERACTIVE)")

# ================================================================
# FINAL SUMMARY
# ================================================================
total_rev  = df['Total_Sales'].sum()
best_prod  = product_rev.idxmax()
best_reg   = region_rev.idxmax()
avg_order  = df['Total_Sales'].mean()
max_sale   = df['Total_Sales'].max()

print("\n" + "=" * 65)
print("         WEEK 6 — SALES DASHBOARD REPORT")
print("=" * 65)
print(f"  Total Revenue        : Rs {total_rev:>12,.0f}")
print(f"  Total Transactions   : {len(df)}")
print(f"  Average Order Value  : Rs {avg_order:>10,.0f}")
print(f"  Best Product         : {best_prod}")
print(f"  Best Region          : {best_reg}")
print(f"  Highest Single Sale  : Rs {max_sale:>10,.0f}")
print(f"  Seaborn Charts (5)   : Bar, Box, Violin, Heatmap, Multi-subplot")
print(f"  Plotly Charts (2)    : Interactive Bar + Full Dashboard")
print(f"  Total Charts         : 7")
print("=" * 65)
print("\nAll charts saved in -> visualizations/")
print("Open .html files in browser for interactive features!")