import pytest
import pandas as pd
import numpy as np
from src.filters import ColumnFilter, InjuryFilter, CompositeFilter


class TestColumnFilter:
    """Tests for ColumnFilter class"""

    def test_greater_than(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3'],
            'salary': [5000, 7000, 9000]
        })

        filter_obj = ColumnFilter('salary', '>', 6000)
        result = filter_obj.apply(data)

        assert len(result) == 2
        assert list(result['playerID']) == ['2', '3']

    def test_greater_than_or_equal(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3'],
            'salary': [5000, 6000, 9000]
        })

        filter_obj = ColumnFilter('salary', '>=', 6000)
        result = filter_obj.apply(data)

        assert len(result) == 2
        assert list(result['playerID']) == ['2', '3']

    def test_less_than(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3'],
            'salary': [5000, 7000, 9000]
        })

        filter_obj = ColumnFilter('salary', '<', 7000)
        result = filter_obj.apply(data)

        assert len(result) == 1
        assert list(result['playerID']) == ['1']

    def test_less_than_or_equal(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3'],
            'salary': [5000, 7000, 9000]
        })

        filter_obj = ColumnFilter('salary', '<=', 7000)
        result = filter_obj.apply(data)

        assert len(result) == 2
        assert list(result['playerID']) == ['1', '2']

    def test_equals(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3'],
            'team': ['LAL', 'BOS', 'LAL']
        })

        filter_obj = ColumnFilter('team', '==', 'LAL')
        result = filter_obj.apply(data)

        assert len(result) == 2
        assert list(result['playerID']) == ['1', '3']

    def test_not_equals(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3'],
            'team': ['LAL', 'BOS', 'LAL']
        })

        filter_obj = ColumnFilter('team', '!=', 'LAL')
        result = filter_obj.apply(data)

        assert len(result) == 1
        assert list(result['playerID']) == ['2']

    def test_in_operator(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3', '4'],
            'pos': ['PG', 'SG', 'SF', 'PF']
        })

        filter_obj = ColumnFilter('pos', 'in', ['PG', 'SG'])
        result = filter_obj.apply(data)

        assert len(result) == 2
        assert list(result['playerID']) == ['1', '2']

    def test_not_in_operator(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3', '4'],
            'pos': ['PG', 'SG', 'SF', 'PF']
        })

        filter_obj = ColumnFilter('pos', 'not_in', ['PG', 'SG'])
        result = filter_obj.apply(data)

        assert len(result) == 2
        assert list(result['playerID']) == ['3', '4']

    def test_invalid_operator(self):
        with pytest.raises(ValueError, match="Invalid operator"):
            ColumnFilter('salary', '><', 5000)

    def test_missing_column(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3'],
            'salary': [5000, 7000, 9000]
        })

        filter_obj = ColumnFilter('team', '==', 'LAL')

        with pytest.raises(ValueError, match="Column 'team' not found"):
            filter_obj.apply(data)

    def test_empty_dataframe(self):
        data = pd.DataFrame()
        filter_obj = ColumnFilter('salary', '>', 5000)
        result = filter_obj.apply(data)

        assert result.empty

    def test_repr(self):
        filter_obj = ColumnFilter('salary', '>', 5000)
        repr_str = repr(filter_obj)

        assert 'ColumnFilter' in repr_str
        assert 'salary' in repr_str
        assert '>' in repr_str
        assert '5000' in repr_str


class TestInjuryFilter:
    """Tests for InjuryFilter class"""

    def test_exclude_out_only(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3', '4'],
            'is_out': [0, 1, 0, 0],
            'is_doubtful': [0, 0, 1, 0],
            'is_questionable': [0, 0, 0, 1]
        })

        filter_obj = InjuryFilter(exclude_out=True)
        result = filter_obj.apply(data)

        assert len(result) == 3
        assert list(result['playerID']) == ['1', '3', '4']

    def test_exclude_doubtful_only(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3', '4'],
            'is_out': [0, 1, 0, 0],
            'is_doubtful': [0, 0, 1, 0],
            'is_questionable': [0, 0, 0, 1]
        })

        filter_obj = InjuryFilter(exclude_out=False, exclude_doubtful=True)
        result = filter_obj.apply(data)

        assert len(result) == 3
        assert list(result['playerID']) == ['1', '2', '4']

    def test_exclude_questionable_only(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3', '4'],
            'is_out': [0, 1, 0, 0],
            'is_doubtful': [0, 0, 1, 0],
            'is_questionable': [0, 0, 0, 1]
        })

        filter_obj = InjuryFilter(exclude_out=False, exclude_questionable=True)
        result = filter_obj.apply(data)

        assert len(result) == 3
        assert list(result['playerID']) == ['1', '2', '3']

    def test_exclude_all(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3', '4'],
            'is_out': [0, 1, 0, 0],
            'is_doubtful': [0, 0, 1, 0],
            'is_questionable': [0, 0, 0, 1]
        })

        filter_obj = InjuryFilter(
            exclude_out=True,
            exclude_doubtful=True,
            exclude_questionable=True
        )
        result = filter_obj.apply(data)

        assert len(result) == 1
        assert list(result['playerID']) == ['1']

    def test_no_exclusions(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3', '4'],
            'is_out': [0, 1, 0, 0],
            'is_doubtful': [0, 0, 1, 0],
            'is_questionable': [0, 0, 0, 1]
        })

        filter_obj = InjuryFilter(
            exclude_out=False,
            exclude_doubtful=False,
            exclude_questionable=False
        )
        result = filter_obj.apply(data)

        assert len(result) == 4

    def test_missing_columns(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3'],
            'salary': [5000, 7000, 9000]
        })

        filter_obj = InjuryFilter(exclude_out=True)

        with pytest.raises(ValueError, match="Required injury columns not found"):
            filter_obj.apply(data)

    def test_empty_dataframe(self):
        data = pd.DataFrame()
        filter_obj = InjuryFilter(exclude_out=True)
        result = filter_obj.apply(data)

        assert result.empty

    def test_repr(self):
        filter_obj = InjuryFilter(exclude_out=True, exclude_doubtful=True)
        repr_str = repr(filter_obj)

        assert 'InjuryFilter' in repr_str
        assert 'exclude_out=True' in repr_str
        assert 'exclude_doubtful=True' in repr_str


class TestCompositeFilter:
    """Tests for CompositeFilter class"""

    def test_and_logic(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3', '4'],
            'salary': [5000, 7000, 9000, 11000],
            'team': ['LAL', 'BOS', 'LAL', 'BOS']
        })

        filters = [
            ColumnFilter('salary', '>', 6000),
            ColumnFilter('team', '==', 'LAL')
        ]

        composite = CompositeFilter(filters, logic='and')
        result = composite.apply(data)

        assert len(result) == 1
        assert list(result['playerID']) == ['3']

    def test_or_logic(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3', '4'],
            'salary': [5000, 7000, 9000, 11000],
            'team': ['LAL', 'BOS', 'GSW', 'GSW']
        })

        filters = [
            ColumnFilter('salary', '<', 6000),
            ColumnFilter('team', '==', 'GSW')
        ]

        composite = CompositeFilter(filters, logic='or')
        result = composite.apply(data)

        assert len(result) == 3
        assert set(result['playerID']) == {'1', '3', '4'}

    def test_three_filters_and(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3', '4', '5'],
            'salary': [5000, 7000, 9000, 11000, 8000],
            'team': ['LAL', 'BOS', 'LAL', 'BOS', 'LAL'],
            'pos': ['PG', 'SG', 'SF', 'PF', 'PG']
        })

        filters = [
            ColumnFilter('salary', '>', 6000),
            ColumnFilter('team', '==', 'LAL'),
            ColumnFilter('pos', '==', 'PG')
        ]

        composite = CompositeFilter(filters, logic='and')
        result = composite.apply(data)

        assert len(result) == 1
        assert list(result['playerID']) == ['5']

    def test_empty_filters_list(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3'],
            'salary': [5000, 7000, 9000]
        })

        composite = CompositeFilter([], logic='and')
        result = composite.apply(data)

        assert len(result) == 3

    def test_invalid_logic(self):
        filters = [ColumnFilter('salary', '>', 5000)]

        with pytest.raises(ValueError, match="Logic must be 'and' or 'or'"):
            CompositeFilter(filters, logic='xor')

    def test_empty_dataframe(self):
        data = pd.DataFrame()
        filters = [ColumnFilter('salary', '>', 5000)]
        composite = CompositeFilter(filters, logic='and')
        result = composite.apply(data)

        assert result.empty

    def test_name_generation(self):
        filters = [
            ColumnFilter('salary', '>', 5000, name='salary_filter'),
            ColumnFilter('team', '==', 'LAL', name='team_filter')
        ]

        composite = CompositeFilter(filters, logic='and')

        assert 'and' in composite.name
        assert 'salary_filter' in composite.name
        assert 'team_filter' in composite.name

    def test_custom_name(self):
        filters = [ColumnFilter('salary', '>', 5000)]
        composite = CompositeFilter(filters, logic='and', name='custom_filter')

        assert composite.name == 'custom_filter'

    def test_repr(self):
        filters = [
            ColumnFilter('salary', '>', 5000, name='salary_filter'),
            ColumnFilter('team', '==', 'LAL', name='team_filter')
        ]

        composite = CompositeFilter(filters, logic='and')
        repr_str = repr(composite)

        assert 'CompositeFilter' in repr_str
        assert 'and' in repr_str
        assert 'salary_filter' in repr_str
        assert 'team_filter' in repr_str

    def test_or_logic_with_exception_handling(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3'],
            'salary': [5000, 7000, 9000]
        })

        filters = [
            ColumnFilter('salary', '>', 6000),
            ColumnFilter('missing_column', '==', 'value')
        ]

        composite = CompositeFilter(filters, logic='or')
        result = composite.apply(data)

        assert len(result) == 2
        assert set(result['playerID']) == {'2', '3'}


class TestFilterIntegration:
    """Integration tests for filter combinations"""

    def test_salary_and_injury_filter(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3', '4'],
            'salary': [5000, 7000, 9000, 11000],
            'is_out': [0, 0, 1, 0],
            'is_doubtful': [0, 0, 0, 0],
            'is_questionable': [0, 0, 0, 0]
        })

        filters = [
            ColumnFilter('salary', '>=', 7000),
            InjuryFilter(exclude_out=True)
        ]

        composite = CompositeFilter(filters, logic='and')
        result = composite.apply(data)

        assert len(result) == 2
        assert set(result['playerID']) == {'2', '4'}

    def test_multiple_salary_bounds(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3', '4', '5'],
            'salary': [4000, 6000, 8000, 10000, 12000]
        })

        filters = [
            ColumnFilter('salary', '>=', 6000),
            ColumnFilter('salary', '<=', 10000)
        ]

        composite = CompositeFilter(filters, logic='and')
        result = composite.apply(data)

        assert len(result) == 3
        assert set(result['playerID']) == {'2', '3', '4'}

    def test_position_team_salary_combination(self):
        data = pd.DataFrame({
            'playerID': ['1', '2', '3', '4', '5'],
            'salary': [5000, 7000, 9000, 11000, 8000],
            'team': ['LAL', 'BOS', 'LAL', 'BOS', 'GSW'],
            'pos': ['PG', 'SG', 'PG', 'PF', 'PG']
        })

        filters = [
            ColumnFilter('salary', '>', 6000),
            ColumnFilter('pos', '==', 'PG'),
            ColumnFilter('team', 'in', ['LAL', 'GSW'])
        ]

        composite = CompositeFilter(filters, logic='and')
        result = composite.apply(data)

        assert len(result) == 2
        assert set(result['playerID']) == {'3', '5'}
