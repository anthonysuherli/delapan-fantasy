1            'num_seasons': args.num_seasons,
            'model_type': args.model_type,
            'feature_config': args.feature_config,
            'per_player_models': args.per_player,
            'recalibrate_days': args.recalibrate_days,
            'n_jobs': args.n_jobs,
            'rewrite_models': args.rewrite_models,
            'model_params': model_params
        }

        report_gen = BacktestReportGenerator(output_path, use_plotly=True)
        run_timestamp = output_path.name

        comprehensive_report_path = report_gen.generate_report(
            results=results,
            config=config,
            run_timestamp=run_timestamp,
            generate_charts=True
        )

        logger.info(f"Comprehensive report generated: {comprehensive_report_path}")
        results['comprehensive_report_path'] = str(comprehensive_report_path)

    except Exception as e:
        logger.error(f"Failed to generate comprehensive report: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    if 'tier_comparison' in results:
        tier_path = output_path / f"tier_comparison_{args.test_start}_to_{args.test_end}.csv"
        results['tier_comparison'].to_csv(tier_path, index=False)
        logger.info(f"Tier comparison saved to: {tier_path}")

    logger.info("")
    logger.info("="*80)
    logger.info("BACKTEST COMPLETE")
    logger.info("="*80)
    logger.info(f"Slates processed: {results.get('num_slates', 0)}")
    logger.info(f"Model MAPE: {results.get('model_mean_mape', 0):.2f}%")
    logger.info(f"Benchmark MAPE: {results.get('benchmark_mean_mape', 0):.2f}%")
    logger.info(f"Improvement: {results.get('mape_improvement', 0):+.2f}%")

    if 'report_path' in results:
        logger.info(f"Comprehensive report: {results['report_path']}")

    if 'comprehensive_report_path' in results:
        logger.info(f"Interactive report: {results['comprehensive_report_path']}")
        charts_dir = output_path / 'charts'
        if charts_dir.exists():
            num_charts = len(list(charts_dir.glob('*.html')))
            logger.info(f"Generated {num_charts} interactive Plotly charts in {charts_dir}")

    logger.info(f"All outputs saved to: {output_path}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
