from sys import stdout


def progress_bar(current: int, total: int, prefix: str = '', suffix: str = '', bar_size=25):
    full_len = int(current * bar_size / total + 0.5)
    empty_len = bar_size - full_len
    if prefix:
        prefix = prefix + ' '
    print('\r\x1b[K', end='')
    print('{}[{}{}] ({}/{}) {}'.format(prefix, '#' * full_len, '.' * empty_len, current, total, suffix), end='')
    if current == total:
        print()
    stdout.flush()


def sec_to_time(sec: int) -> str:
    """ Convert second to human readable time """
    h = sec // 3600
    sec = sec % 3600
    m = sec // 60
    sec = sec % 60
    return '{:02}:{:02}:{:02}'.format(h, m, sec)
