from prometheus_client import Histogram

function_duration_seconds = Histogram(
    'function_duration_seconds',
    'Time spent in function',
    ['function_name']
)