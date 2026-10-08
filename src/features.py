import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray

DATE_COLUMNS = ['cookie_created_at', 'window_start_ts', 'window_end_ts']
PLATFORM_MAP = {
    'web': 'web',
    'desktop': 'desktop',
    'android': 'android',
    'ios': 'ios',
    'iphone': 'ios',
}
EVENT_NAMES = [
    'item_view',
    'search_results_view',
    'photo_swipe',
    'favorite_add',
    'seller_page_view',
    'contact_phone_show',
    'contact_chat_open',
    'contact_message_sent',
    'login',
]
AUTOMATION_LIBRARIES = (
    r'^(python-requests|python-urllib3|curl|Go-http-client|node-fetch|Scrapy)/'
)
AUTOMATION_FAMILIES = [
    'python-requests',
    'python-urllib3',
    'curl',
    'Go-http-client',
    'node-fetch',
    'Scrapy',
    'headless_chrome',
]
BROWSER_RULES = {
    'headless_chrome': r'HeadlessChrome/(\d+)',
    'yandex_browser': r'YaBrowser/23\.(\d+)',
    'firefox': r'Firefox/(\d+)',
    'native_app': r'^(?!Mozilla)[A-Za-z]+/(\d+)\.\d+ \((?:Android|iPhone)',
    'safari': r'Version/(\d+)',
    'chrome': r'Chrome/(\d+)',
}
OS_PATTERNS = {
    'windows': 'Windows',
    'macos': 'Macintosh',
    'android': 'Android',
    'ios': 'iPhone|iOS',
    'linux': 'Linux',
}
NIGHT_HOURS = range(6)
DEEP_SEARCH_PAGE = 3
CATEGORICAL_FEATURES = [
    'platform_combination',
    'ua_kind',
    'browser_family',
    'os_name',
    'main_category',
]


def read_cookies(cookies_path: str) -> pd.DataFrame:
    return pd.read_csv(cookies_path, parse_dates=DATE_COLUMNS)


def read_events(events_path: str) -> pd.DataFrame:
    return pd.read_csv(events_path, parse_dates=['event_ts'])


def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cookies = read_cookies(f'{data_dir}/train.csv')
    events = read_events(f'{data_dir}/events.csv.gz')
    return cookies, events


def parse_user_agents(
    user_agents: np.ndarray | ExtensionArray,
) -> pd.DataFrame:
    parsed = pd.DataFrame({'user_agent': user_agents})
    parsed['browser_family'] = parsed.user_agent.str.extract(
        AUTOMATION_LIBRARIES
    )[0]
    parsed['browser_version'] = np.nan
    for family, pattern in BROWSER_RULES.items():
        version = parsed.user_agent.str.extract(pattern)[0].astype(float)
        is_new_match = parsed.browser_family.isna() & version.notna()
        parsed.loc[is_new_match, 'browser_family'] = family
        parsed.loc[is_new_match, 'browser_version'] = version[is_new_match]
    parsed['os_name'] = np.select(
        [
            parsed.user_agent.str.contains(pattern)
            for pattern in OS_PATTERNS.values()
        ],
        list(OS_PATTERNS),
        'unknown',
    )
    parsed['os_version'] = parsed.user_agent.str.extract(
        r'(?:Android |iPhone OS |iOS )(\d+)'
    )[0].astype(float)
    ua_kind_conditions = [
        parsed.browser_family.isin(AUTOMATION_FAMILIES),
        parsed.browser_family == 'native_app',
        parsed.os_name.isin(['android', 'ios']),
    ]
    parsed['ua_kind'] = np.select(
        ua_kind_conditions,
        ['automation', 'native_app', 'mobile_browser'],
        'desktop_browser',
    )
    return parsed.set_index('user_agent')


def select_window_events(
    events: pd.DataFrame, cookies: pd.DataFrame
) -> pd.DataFrame:
    window_events = events.merge(
        cookies[['cookie_id', 'window_start_ts', 'window_end_ts']],
        on='cookie_id',
    )
    window_events = window_events[
        (window_events.event_ts >= window_events.window_start_ts)
        & (window_events.event_ts < window_events.window_end_ts)
    ].copy()
    window_events['platform'] = (
        window_events.platform.str.strip().str.lower().map(PLATFORM_MAP)
    )
    window_events = window_events.join(
        parse_user_agents(window_events.user_agent.unique()), on='user_agent'
    )
    window_events = window_events.sort_values(
        ['cookie_id', 'event_ts']
    ).reset_index(drop=True)
    window_events['hour'] = window_events.event_ts.dt.hour
    window_events['gap_seconds'] = (
        window_events.groupby('cookie_id').event_ts.diff().dt.total_seconds()
    )
    window_events['is_night'] = window_events.hour.isin(NIGHT_HOURS)
    window_events['is_zero_second'] = window_events.event_ts.dt.second == 0
    window_events['is_same_ts_as_previous'] = window_events.gap_seconds == 0
    window_events['is_automation_ua'] = window_events.ua_kind == 'automation'
    return window_events


def time_features(window_events: pd.DataFrame) -> pd.DataFrame:
    by_cookie = window_events.groupby('cookie_id')
    features = by_cookie.agg(
        gap_median_seconds=('gap_seconds', 'median'),
        gap_min_seconds=('gap_seconds', 'min'),
        gap_mean_seconds=('gap_seconds', 'mean'),
        gap_std_seconds=('gap_seconds', 'std'),
        max_gap_seconds=('gap_seconds', 'max'),
    )
    features['gap_cv'] = features.gap_std_seconds / features.gap_mean_seconds
    features['max_gap_hours'] = features.pop('max_gap_seconds') / 3600
    features['night_event_share'] = by_cookie.is_night.mean()
    features['zero_second_share'] = by_cookie.is_zero_second.mean()
    features['same_ts_share'] = by_cookie.is_same_ts_as_previous.mean()
    features['active_hours'] = by_cookie.hour.nunique()
    features['session_span_hours'] = (
        by_cookie.event_ts.max() - by_cookie.event_ts.min()
    ).dt.total_seconds() / 3600
    hour_counts = pd.crosstab(window_events.cookie_id, window_events.hour)
    hour_probabilities = hour_counts.div(hour_counts.sum(axis=1), axis=0)
    features['hour_entropy'] = -(
        hour_probabilities
        * np.log2(hour_probabilities.where(hour_probabilities > 0))
    ).sum(axis=1)
    max_possible_entropy = (
        hour_counts.sum(axis=1).clip(upper=24).map(np.log2).replace(0, np.nan)
    )
    features['hour_entropy_normalized'] = (
        features.hour_entropy / max_possible_entropy
    )
    return features


def user_agent_features(window_events: pd.DataFrame) -> pd.DataFrame:
    by_cookie = window_events.groupby('cookie_id')
    features = pd.DataFrame(
        {
            'n_user_agents': by_cookie.user_agent.nunique(),
            'automation_event_share': by_cookie.is_automation_ua.mean(),
        }
    )
    features['platform_combination'] = by_cookie.platform.agg(
        lambda platforms: ' + '.join(sorted(platforms.unique()))
    )
    user_agent_counts = (
        window_events.groupby(['cookie_id', 'user_agent'])
        .size()
        .rename('n_events')
        .reset_index()
    )
    main_user_agent = (
        user_agent_counts.sort_values(
            ['cookie_id', 'n_events'], ascending=[True, False]
        )
        .drop_duplicates('cookie_id')
        .set_index('cookie_id')
        .user_agent
    )
    main_browser = (
        parse_user_agents(main_user_agent.unique())
        .reindex(main_user_agent.to_list())
        .set_axis(main_user_agent.index)
    )
    return features.join(main_browser)


def content_features(window_events: pd.DataFrame) -> pd.DataFrame:
    by_cookie = window_events.groupby('cookie_id')
    item_events = window_events[window_events.item_id.notna()]
    item_views = window_events[window_events.event_name == 'item_view']
    search_events = window_events[
        window_events.event_name == 'search_results_view'
    ].copy()
    contact_events = window_events[
        window_events.event_name.str.startswith('contact_')
    ]
    features = pd.DataFrame(
        {
            'n_unique_categories': by_cookie.item_category.nunique(),
            'n_unique_locations': by_cookie.item_location.nunique(),
        }
    )
    features['main_category'] = (
        window_events.dropna(subset=['item_category'])
        .groupby('cookie_id')
        .item_category.agg(
            lambda categories: categories.value_counts().index[0]
        )
    )
    features['n_item_events'] = item_events.groupby('cookie_id').size()
    features['n_unique_items'] = item_events.groupby(
        'cookie_id'
    ).item_id.nunique()
    features['events_per_item'] = (
        features.n_item_events / features.n_unique_items
    )
    features['repeat_view_share'] = (
        item_views.duplicated(['cookie_id', 'item_id'])
        .groupby(item_views.cookie_id)
        .mean()
    )
    items_with_photos = window_events.loc[
        window_events.event_name == 'photo_swipe', ['cookie_id', 'item_id']
    ].drop_duplicates()
    viewed_items = (
        item_views[['cookie_id', 'item_id']]
        .drop_duplicates()
        .merge(items_with_photos, how='left', indicator='photo_swipe_match')
    )
    features['viewed_items_with_photo_swipe_share'] = (
        viewed_items.photo_swipe_match.eq('both')
        .groupby(viewed_items.cookie_id)
        .mean()
    )
    features['pro_seller_share'] = (
        item_events.seller_type.eq('pro').groupby(item_events.cookie_id).mean()
    )
    contact_counts = (
        contact_events.groupby('cookie_id')
        .size()
        .reindex(features.index, fill_value=0)
    )
    features['contacts_per_item_view'] = (
        contact_counts / item_views.groupby('cookie_id').size()
    )
    first_item_view_ts = (
        item_views.groupby(['cookie_id', 'item_id'])
        .event_ts.min()
        .rename('first_item_view_ts')
    )
    contacts_with_view = contact_events.join(
        first_item_view_ts, on=['cookie_id', 'item_id']
    )
    features['contact_without_view_share'] = (
        (
            ~contacts_with_view.first_item_view_ts.le(
                contacts_with_view.event_ts
            )
        )
        .groupby(contacts_with_view.cookie_id)
        .mean()
    )
    search_by_cookie = search_events.groupby('cookie_id')
    search_events['is_next_page'] = (
        search_events.search_page == search_by_cookie.search_page.shift() + 1
    ) & (search_events.search_query == search_by_cookie.search_query.shift())
    search_events['is_deep_page'] = (
        search_events.search_page > DEEP_SEARCH_PAGE
    )
    search_features = search_events.groupby('cookie_id').agg(
        n_search_events=('search_page', 'size'),
        n_unique_queries=('search_query', 'nunique'),
        max_search_page=('search_page', 'max'),
        deep_page_share=('is_deep_page', 'mean'),
        next_page_share=('is_next_page', 'mean'),
    )
    search_features['searches_per_query'] = (
        search_features.n_search_events / search_features.n_unique_queries
    )
    return features.join(search_features)


def pointer_features(window_events: pd.DataFrame) -> pd.DataFrame:
    web_desktop_events = window_events[
        window_events.platform.isin(['desktop', 'web'])
    ]
    pointer_missing_share = (
        web_desktop_events.pointer_x.isna()
        .groupby(web_desktop_events.cookie_id)
        .mean()
        .rename('pointer_missing_share')
    )
    pointer_events = window_events[window_events.pointer_x.notna()].copy()
    is_edge_x = (pointer_events.pointer_x <= 0) | (
        pointer_events.pointer_x >= 1919
    )
    is_edge_y = (pointer_events.pointer_y <= 0) | (
        pointer_events.pointer_y >= 1080
    )
    pointer_events['is_screen_edge'] = is_edge_x | is_edge_y
    pointer_events['is_round_coordinate'] = (
        pointer_events.pointer_x % 50 == 0
    ) & (pointer_events.pointer_y % 50 == 0)
    pointer_events['is_repeated_point'] = pointer_events.duplicated(
        ['cookie_id', 'pointer_x', 'pointer_y']
    )
    pointer_aggregations = {
        'n_pointer_events': ('pointer_x', 'size'),
        'pointer_x_mean': ('pointer_x', 'mean'),
        'pointer_x_std': ('pointer_x', 'std'),
        'pointer_y_std': ('pointer_y', 'std'),
        'screen_edge_share': ('is_screen_edge', 'mean'),
        'round_coordinate_share': ('is_round_coordinate', 'mean'),
        'repeated_point_share': ('is_repeated_point', 'mean'),
    }
    return (
        pointer_events.groupby('cookie_id')
        .agg(**pointer_aggregations)
        .join(pointer_missing_share, how='outer')
    )


def build_features(
    events: pd.DataFrame, cookies: pd.DataFrame
) -> pd.DataFrame:
    window_events = select_window_events(events, cookies)
    cookie_meta = cookies.set_index('cookie_id')
    features = pd.DataFrame(index=cookie_meta.index)
    features['cookie_age_days'] = (
        cookie_meta.window_start_ts - cookie_meta.cookie_created_at
    ).dt.total_seconds() / 86400
    features['n_events'] = window_events.groupby('cookie_id').size()
    event_type_share = pd.crosstab(
        window_events.cookie_id, window_events.event_name, normalize='index'
    ).reindex(columns=EVENT_NAMES, fill_value=0)
    features = features.join(event_type_share.add_suffix('_share'))
    features['login_count'] = (
        window_events.event_name.eq('login')
        .groupby(window_events.cookie_id)
        .sum()
    )
    for feature_block in [
        time_features,
        user_agent_features,
        content_features,
        pointer_features,
    ]:
        features = features.join(feature_block(window_events))
    features['n_pointer_events'] = features.n_pointer_events.fillna(0)
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype('category')
    return features
