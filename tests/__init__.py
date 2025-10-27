graph TB
    subgraph Data["Data Layer"]
        A[Tank01 API] --> B[Cache]
        B --> C[Parquet Storage]
        C --> D[Historical Loader]
    end

    subgraph Feature["Feature Layer"]
        D --> E[YAML Config]
        E --> F[Feature Pipeline]
        F --> G[Rolling Stats]
        F --> H[EWMA]
        G --> I[147 Features]
        H --> I
    end

    subgraph Model["Model Layer"]
        I --> J[Per-Player XGBoost]
        J --> K[Model Registry]
        K --> L[Saved Models]
    end

    subgraph Optimization["Optimization Layer"]
        J --> M[Projections]
        M --> N[Linear Programming]
        N --> O[DraftKings Constraints]
        O --> P[Optimal Lineups]
    end

    subgraph Evaluation["Evaluation Layer"]
        M --> Q[Walk-Forward Backtest]
        Q --> R[Metrics: MAPE/RMSE/MAE]
        Q --> S[Benchmark Comparison]
        R --> T[Results by Salary Tier]
    end
