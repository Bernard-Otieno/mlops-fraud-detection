import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/processed/my_features.csv')

print("="*70)
print("📊 M-PESA SMS FRAUD DETECTION - VISUALIZATION")
print("="*70)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Create a 3x3 grid of visualizations
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('M-PESA SMS Fraud Detection - Feature Analysis', 
             fontsize=16, fontweight='bold', y=0.995)

# ========================================
# Visualization 1: Overall Fraud Distribution
# ========================================
ax = axes[0, 0]
fraud_counts = df['is_fraud'].value_counts()
colors = ['#2ecc71', '#e74c3c']
bars = ax.bar(['Legitimate', 'Fraud'], fraud_counts.values, color=colors, 
              alpha=0.8, edgecolor='black', linewidth=2)
ax.set_title('Overall Distribution', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=10)

# Add count labels on bars
for bar, count in zip(bars, fraud_counts.values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count:,}\n({count/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold')

# ========================================
# Visualization 2: Valid Sender vs Fraud
# ========================================
ax = axes[0, 1]
sender_fraud = df.groupby(['is_valid_sender', 'is_fraud']).size().unstack(fill_value=0)
sender_fraud.plot(kind='bar', ax=ax, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=2)
ax.set_title('Valid Sender ID Impact', fontsize=12, fontweight='bold')
ax.set_xlabel('Is Valid Sender (MPESA)', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_xticklabels(['Invalid', 'Valid'], rotation=0)
ax.legend(['Legitimate', 'Fraud'], loc='upper right')

# Calculate fraud rates
for i in [0, 1]:
    subset = df[df['is_valid_sender'] == i]
    fraud_rate = subset['is_fraud'].mean() * 100
    ax.text(i, subset.shape[0] * 0.5, f'{fraud_rate:.0f}%\nfraud',
            ha='center', fontsize=9, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# ========================================
# Visualization 3: Message Length Distribution
# ========================================
ax = axes[0, 2]
legit_length = df[df['is_fraud']==0]['message_length']
fraud_length = df[df['is_fraud']==1]['message_length']

ax.hist([legit_length, fraud_length], bins=30, label=['Legitimate', 'Fraud'],
        color=colors, alpha=0.7, edgecolor='black')
ax.set_title('Message Length Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Message Length (characters)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.legend()
ax.axvline(legit_length.mean(), color='green', linestyle='--', linewidth=2, 
           label=f'Legit avg: {legit_length.mean():.0f}')
ax.axvline(fraud_length.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Fraud avg: {fraud_length.mean():.0f}')

# ========================================
# Visualization 4: PIN Request Analysis
# ========================================
ax = axes[1, 0]
pin_fraud = df.groupby(['has_pin_word', 'is_fraud']).size().unstack(fill_value=0)
pin_fraud.plot(kind='bar', ax=ax, color=colors, alpha=0.8, 
               edgecolor='black', linewidth=2)
ax.set_title('PIN Request Impact', fontsize=12, fontweight='bold')
ax.set_xlabel('Contains PIN Word', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_xticklabels(['No', 'Yes'], rotation=0)
ax.legend(['Legitimate', 'Fraud'])

# Add fraud rates
for i in [0, 1]:
    subset = df[df['has_pin_word'] == i]
    if len(subset) > 0:
        fraud_rate = subset['is_fraud'].mean() * 100
        ax.text(i, subset.shape[0] * 0.5, f'{fraud_rate:.0f}%',
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# ========================================
# Visualization 5: Urgent Words Analysis
# ========================================
ax = axes[1, 1]
urgent_fraud = df.groupby(['has_urgent_word', 'is_fraud']).size().unstack(fill_value=0)
urgent_fraud.plot(kind='bar', ax=ax, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=2)
ax.set_title('Urgency Keywords Impact', fontsize=12, fontweight='bold')
ax.set_xlabel('Contains Urgent Words', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_xticklabels(['No', 'Yes'], rotation=0)
ax.legend(['Legitimate', 'Fraud'])

# Add fraud rates
for i in [0, 1]:
    subset = df[df['has_urgent_word'] == i]
    if len(subset) > 0:
        fraud_rate = subset['is_fraud'].mean() * 100
        ax.text(i, subset.shape[0] * 0.5, f'{fraud_rate:.0f}%',
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# ========================================
# Visualization 6: Link Presence Analysis
# ========================================
ax = axes[1, 2]
link_fraud = df.groupby(['has_link', 'is_fraud']).size().unstack(fill_value=0)
link_fraud.plot(kind='bar', ax=ax, color=colors, alpha=0.8,
                edgecolor='black', linewidth=2)
ax.set_title('Link/URL Presence Impact', fontsize=12, fontweight='bold')
ax.set_xlabel('Contains Link', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_xticklabels(['No', 'Yes'], rotation=0)
ax.legend(['Legitimate', 'Fraud'])

# ========================================
# Visualization 7: Exclamation Mark Usage
# ========================================
ax = axes[2, 0]
legit_exclaim = df[df['is_fraud']==0]['exclamation_count']
fraud_exclaim = df[df['is_fraud']==1]['exclamation_count']

ax.hist([legit_exclaim, fraud_exclaim], bins=range(0, 8), 
        label=['Legitimate', 'Fraud'], color=colors, alpha=0.7, 
        edgecolor='black', align='left')
ax.set_title('Exclamation Mark Usage', fontsize=12, fontweight='bold')
ax.set_xlabel('Number of "!" in Message', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.legend()
ax.text(0.98, 0.97, f'Avg Legit: {legit_exclaim.mean():.2f}\nAvg Fraud: {fraud_exclaim.mean():.2f}',
        transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ========================================
# Visualization 8: Feature Correlation Heatmap
# ========================================
ax = axes[2, 1]
features = ['is_valid_sender', 'message_length', 'has_pin_word', 
            'amount_count', 'has_confirmed', 'has_link', 
            'exclamation_count', 'has_urgent_word', 'is_fraud']

corr_matrix = df[features].corr()
im = ax.imshow(corr_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

# Add labels
ax.set_xticks(range(len(features)))
ax.set_yticks(range(len(features)))
ax.set_xticklabels([f.replace('_', '\n') for f in features], 
                    rotation=45, ha='right', fontsize=8)
ax.set_yticklabels([f.replace('_', '\n') for f in features], fontsize=8)
ax.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Correlation', rotation=270, labelpad=15)

# ========================================
# Visualization 9: Top Features Bar Chart
# ========================================
ax = axes[2, 2]
all_features = ['is_valid_sender', 'message_length', 'has_pin_word', 
                'amount_count', 'has_confirmed', 'has_link', 
                'exclamation_count', 'has_urgent_word']

correlations = []
for feature in all_features:
    corr = df[[feature, 'is_fraud']].corr().iloc[0, 1]
    correlations.append((feature, corr))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)
feature_names = [c[0].replace('_', '\n') for c, _ in correlations]
corr_values = [c for _, c in correlations]

bar_colors = ['red' if c > 0 else 'green' for c in corr_values]
bars = ax.barh(feature_names, corr_values, color=bar_colors, alpha=0.7, 
               edgecolor='black', linewidth=2)

ax.set_title('Feature Importance (Correlation with Fraud)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Correlation Coefficient', fontsize=10)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, corr_values)):
    ax.text(val + (0.02 if val > 0 else -0.02), i, f'{val:+.3f}',
            va='center', ha='left' if val > 0 else 'right',
            fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('mpesa_fraud_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved visualization to: mpesa_fraud_analysis.png")
plt.savefig('mpesa_fraud_analysis.png', dpi=300, bbox_inches='tight')   
plt.show()

# ========================================
# Print Summary Statistics
# ========================================
print("\n" + "="*70)
print("📊 SUMMARY STATISTICS")
print("="*70)

print(f"\n📈 Dataset Overview:")
print(f"   Total messages: {len(df):,}")
print(f"   Fraud: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")
print(f"   Legitimate: {(~df['is_fraud'].astype(bool)).sum():,} ({(~df['is_fraud'].astype(bool)).mean()*100:.1f}%)")

print(f"\n🎯 Key Feature Insights:")

# Valid sender
invalid_sender = df[df['is_valid_sender']==0]
print(f"\n1. Invalid Sender ID:")
print(f"   Messages: {len(invalid_sender):,}")
print(f"   Fraud rate: {invalid_sender['is_fraud'].mean()*100:.1f}%")
print(f"   Risk: {invalid_sender['is_fraud'].mean() / df['is_fraud'].mean():.1f}x baseline")

# PIN requests
pin_msgs = df[df['has_pin_word']==1]
if len(pin_msgs) > 0:
    print(f"\n2. PIN Requests:")
    print(f"   Messages: {len(pin_msgs):,}")
    print(f"   Fraud rate: {pin_msgs['is_fraud'].mean()*100:.1f}%")
    print(f"   Risk: {pin_msgs['is_fraud'].mean() / df['is_fraud'].mean():.1f}x baseline")

# Urgent words
urgent_msgs = df[df['has_urgent_word']==1]
if len(urgent_msgs) > 0:
    print(f"\n3. Urgent Language:")
    print(f"   Messages: {len(urgent_msgs):,}")
    print(f"   Fraud rate: {urgent_msgs['is_fraud'].mean()*100:.1f}%")
    print(f"   Risk: {urgent_msgs['is_fraud'].mean() / df['is_fraud'].mean():.1f}x baseline")

# Links
link_msgs = df[df['has_link']==1]
if len(link_msgs) > 0:
    print(f"\n4. Contains Links:")
    print(f"   Messages: {len(link_msgs):,}")
    print(f"   Fraud rate: {link_msgs['is_fraud'].mean()*100:.1f}%")
    print(f"   Risk: {link_msgs['is_fraud'].mean() / df['is_fraud'].mean():.1f}x baseline")

print("\n" + "="*70)
print("✨ EDA Complete!")
print("="*70)