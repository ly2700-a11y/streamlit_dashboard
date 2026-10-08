# Regime Detection & A2C Integration Diagrams

## 1. Regime Detection - FFT Extraction Process

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#000000', 'primaryTextColor': '#FFFFFF', 'primaryBorderColor': '#000000', 'backgroundColor': '#FFFFFF', 'secondaryColor': '#000000', 'secondaryTextColor': '#FFFFFF', 'secondaryBorderColor': '#000000', 'tertiaryColor': '#000000', 'tertiaryTextColor': '#FFFFFF', 'tertiaryBorderColor': '#000000', 'noteBkgColor': '#000000', 'noteBorderColor': '#000000', 'noteTextColor': '#FFFFFF'}}}%%
graph LR
    A["📊 Load CLC Data<br/>50 contracts"] --> B["🔧 Build State Matrices<br/>For each asset at each time point<br/>Construct 60×9 matrix"]
    B --> C["📈 Extract FFT Features<br/>9 columns independent FFT →<br/>10 Real + 10 Imaginary<br/>= 180-dimensional vector"]
    C --> D["🎲 GMM Clustering<br/>n_components=3<br/>Fit Gaussian Mixture Model"]
    D --> E["🏷️ Output Regime"]
    E --> E1["Hard Labels<br/>predict()<br/>Each time point →<br/>0/1/2"]
    E --> E2["Soft Probabilities<br/>predict_proba()<br/>Each time point →<br/>P=[p0,p1,p2]"]
    
    style A fill:#000000,stroke:#000000,color:#FFFFFF
    style B fill:#000000,stroke:#000000,color:#FFFFFF
    style C fill:#000000,stroke:#000000,color:#FFFFFF
    style D fill:#000000,stroke:#000000,color:#FFFFFF
    style E fill:#000000,stroke:#000000,color:#FFFFFF
    style E1 fill:#000000,stroke:#000000,color:#FFFFFF
    style E2 fill:#000000,stroke:#000000,color:#FFFFFF
```

---

## 2. Route A - Regime-Specific Ensemble (3 Independent A2C Models)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#000000', 'primaryTextColor': '#FFFFFF', 'primaryBorderColor': '#000000', 'backgroundColor': '#FFFFFF', 'secondaryColor': '#000000', 'secondaryTextColor': '#FFFFFF', 'secondaryBorderColor': '#000000', 'tertiaryColor': '#000000', 'tertiaryTextColor': '#FFFFFF', 'tertiaryBorderColor': '#000000', 'noteBkgColor': '#000000', 'noteBorderColor': '#000000', 'noteTextColor': '#FFFFFF'}}}%%
graph TD
    subgraph train["🔴 TRAINING PHASE"]
        A1["Get Hard Regime Labels<br/>predict() → 0/1/2"] --> A2["Split training data by regime"]
        A2 --> A3["Regime=0 samples<br/>~1/3 of data"]
        A2 --> A4["Regime=1 samples<br/>~1/3 of data"]
        A2 --> A5["Regime=2 samples<br/>~1/3 of data"]
        A3 --> A6["Train A2C_0"]
        A4 --> A7["Train A2C_1"]
        A5 --> A8["Train A2C_2"]
    end
    
    subgraph test["🟢 TEST PHASE"]
        B1["Load test data"] --> B2["Get Regime Soft Probabilities<br/>predict_proba()<br/>→ [p0,p1,p2]"]
        B2 --> B3["Parallel inference"]
        B3 --> B3A["A2C_0 inference<br/>→ action_0"]
        B3 --> B3B["A2C_1 inference<br/>→ action_1"]
        B3 --> B3C["A2C_2 inference<br/>→ action_2"]
        B3A --> B4["Weighted combination<br/>final_action =<br/>a0×p0 + a1×p1 + a2×p2"]
        B3B --> B4
        B3C --> B4
        B4 --> B5["Calculate PnL"]
    end
    
    A6 --> B1
    A7 --> B1
    A8 --> B1
    
    style train fill:#000000,stroke:#000000,color:#FFFFFF
    style test fill:#000000,stroke:#000000,color:#FFFFFF
    style A1 fill:#000000,stroke:#000000,color:#FFFFFF
    style A2 fill:#000000,stroke:#000000,color:#FFFFFF
    style A3 fill:#000000,stroke:#000000,color:#FFFFFF
    style A4 fill:#000000,stroke:#000000,color:#FFFFFF
    style A5 fill:#000000,stroke:#000000,color:#FFFFFF
    style A6 fill:#000000,stroke:#000000,color:#FFFFFF
    style A7 fill:#000000,stroke:#000000,color:#FFFFFF
    style A8 fill:#000000,stroke:#000000,color:#FFFFFF
    style B1 fill:#000000,stroke:#000000,color:#FFFFFF
    style B2 fill:#000000,stroke:#000000,color:#FFFFFF
    style B3 fill:#000000,stroke:#000000,color:#FFFFFF
    style B3A fill:#000000,stroke:#000000,color:#FFFFFF
    style B3B fill:#000000,stroke:#000000,color:#FFFFFF
    style B3C fill:#000000,stroke:#000000,color:#FFFFFF
    style B4 fill:#000000,stroke:#000000,color:#FFFFFF
    style B5 fill:#000000,stroke:#000000,color:#FFFFFF
```

---

## 3. Route B - Regime as State Feature (Single A2C, Recommended ⭐)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#000000', 'primaryTextColor': '#FFFFFF', 'primaryBorderColor': '#000000', 'backgroundColor': '#FFFFFF', 'secondaryColor': '#000000', 'secondaryTextColor': '#FFFFFF', 'secondaryBorderColor': '#000000', 'tertiaryColor': '#000000', 'tertiaryTextColor': '#FFFFFF', 'tertiaryBorderColor': '#000000', 'noteBkgColor': '#000000', 'noteBorderColor': '#000000', 'noteTextColor': '#FFFFFF'}}}%%
graph LR
    subgraph train["🔴 TRAINING PHASE"]
        direction LR
        A1["Get Regime Soft Probabilities<br/>predict_proba()<br/>→ [p0,p1,p2]"] --> A2["Augment State Features"]
        A2 --> A3["Original State Features<br/>9-dim: price, MACD, RSI, etc."]
        A1 --> A4["Regime Features<br/>3-dim: P0, P1, P2"]
        A3 --> A5["Concatenate<br/>Augmented State<br/>12-dim"]
        A4 --> A5
        A5 --> A6["Train Single A2C<br/>with full training data"]
        A6 --> A7["LSTM learns to use<br/>regime information"]
    end
    
    subgraph test["🟢 TEST PHASE"]
        direction LR
        B1["Load test data"] --> B2["Get Augmented State<br/>[Original 9-dim + Soft prob 3-dim]<br/>= 12-dim input"]
        B2 --> B3["Single A2C inference<br/>LSTM automatically handles<br/>regime conditioning"]
        B3 --> B4["Output action directly"]
        B4 --> B5["Calculate PnL"]
    end
    
    A7 --> B1
    
    style train fill:#FFFFFF,stroke:#000000,color:#000000
    style test fill:#FFFFFF,stroke:#000000,color:#000000
    style A1 fill:#000000,stroke:#000000,color:#FFFFFF
    style A2 fill:#000000,stroke:#000000,color:#FFFFFF
    style A3 fill:#000000,stroke:#000000,color:#FFFFFF
    style A4 fill:#000000,stroke:#000000,color:#FFFFFF
    style A5 fill:#000000,stroke:#000000,color:#FFFFFF
    style A6 fill:#000000,stroke:#000000,color:#FFFFFF
    style A7 fill:#000000,stroke:#000000,color:#FFFFFF
    style B1 fill:#000000,stroke:#000000,color:#FFFFFF
    style B2 fill:#000000,stroke:#000000,color:#FFFFFF
    style B3 fill:#000000,stroke:#000000,color:#FFFFFF
    style B4 fill:#000000,stroke:#000000,color:#FFFFFF
    style B5 fill:#000000,stroke:#000000,color:#FFFFFF
```

---

## 4. Comparison: Two Routes

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#000000', 'primaryTextColor': '#FFFFFF', 'primaryBorderColor': '#000000', 'backgroundColor': '#FFFFFF', 'secondaryColor': '#000000', 'secondaryTextColor': '#FFFFFF', 'secondaryBorderColor': '#000000', 'tertiaryColor': '#000000', 'tertiaryTextColor': '#FFFFFF', 'tertiaryBorderColor': '#000000', 'noteBkgColor': '#000000', 'noteBorderColor': '#000000', 'noteTextColor': '#FFFFFF'}}}%%
graph LR
    A["🔵 Route A<br/>Regime-Specific<br/>Ensemble<br/>3 A2C Models"] 
    B["🟦 Route B<br/>Regime as<br/>Feature<br/>1 A2C Model<br/>Recommended ⭐"]
    
    A --> A1["✅ Pros<br/>• Visible regime differences<br/>• Clear structure"]
    A --> A2["❌ Cons<br/>• Severe data shortage<br/>• High underfitting risk<br/>• Ambiguous aggregation<br/>• High complexity"]
    
    B --> B1["✅ Pros<br/>• Uses full training data<br/>• High stability<br/>• Simple implementation<br/>• LSTM-native support<br/>• Usually better results"]
    B --> B2["❌ Cons<br/>• Cannot see explicit<br/>  regime differences"]
    
    style A fill:#000000,stroke:#000000,color:#FFFFFF
    style B fill:#000000,stroke:#000000,color:#FFFFFF
    style A1 fill:#000000,stroke:#000000,color:#FFFFFF
    style A2 fill:#000000,stroke:#000000,color:#FFFFFF
    style B1 fill:#000000,stroke:#000000,color:#FFFFFF
    style B2 fill:#000000,stroke:#000000,color:#FFFFFF
```

---

## Summary

- **Diagram 1**: Shows how regime labels and probabilities are extracted from CLC data using FFT + GMM
- **Diagram 2 (Route A)**: Regime-specific ensemble approach - 3 separate A2C models, one per regime
- **Diagram 3 (Route B)**: Recommended approach - Single A2C with regime probabilities as augmented state features
- **Diagram 4**: Comparison of pros and cons for both routes

**Recommendation**: Start with **Route B** for better data efficiency and stability.

---

To view these diagrams:
1. Use VS Code Mermaid extension: https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid
2. Or export to SVG/PNG using: https://mermaid.live/
3. Copy the markdown code blocks and paste into Mermaid Live editor
