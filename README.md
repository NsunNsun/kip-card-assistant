# kip-card-assistant

Прототип AI-помощника контент-менеджера B2B-магазина КИПиА. По данным о приборе
(название, производитель, характеристики, описание) готовит черновик карточки товара:

- SEO-заголовок до 70 символов;
- SEO-описание на 800–1200 символов;
- список ключевых характеристик, **дословно из входных данных**, отсутствующие помечены «данных нет»;
- 3–5 поисковых запросов;
- блок для менеджера: кому подходит прибор и какие задачи решает;
- список того, чего не хватает во входных данных и что проверить.

Модель пишет только тексты. Характеристики собирает код, а каждое число и технический код
в тексте модели сверяется со входными данными. Второй запрос к модели («судья») проверяет
смысл: не перенесено ли свойство части на весь прибор, нет ли оценок и сфер применения,
которых нет в данных. Промпт — версия 3, доработанная по итогам
двух прогонов на реальной модели (что нашли и что исправили — в разделе 10 DECISIONS). Подробнее — в [docs/DECISIONS.md](docs/DECISIONS.md).

## Быстрый старт

Нужен Python 3.10+ и ключ [OpenRouter](https://openrouter.ai) (или любой OpenAI-совместимый сервер,
в том числе локальный — см. ниже).

```bash
git clone https://github.com/NsunNsun/kip-card-assistant.git
cd kip-card-assistant
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
# откройте .env и впишите LLM_API_KEY; модель — LLM_MODEL
```

Сгенерировать карточки для примеров:

```bash
python -m kip_card generate examples/input/*.json -o examples/output
```

В Windows (PowerShell, cmd) команда та же: шаблон `*.json` программа раскрывает сама,
если оболочка этого не сделала.

```powershell
python -m kip_card generate examples\input\*.json -o examples\output
```

Открыть веб-форму:

```bash
python -m kip_card serve           # http://127.0.0.1:8000
```

Проверить входные данные **без модели и без затрат** (что отсутствует, что подозрительно):

```bash
python -m kip_card check examples/input/manometer-dm02-from-task.json
```

## Примеры

| Вход | Результат | Что показывает |
|---|---|---|
| [testo-830-t1.json](examples/input/testo-830-t1.json) | [md](examples/output/testo-830-t1.md) · [json](examples/output/testo-830-t1.json) | Переносной пирометр, полные данные со страницы товара |
| [owen-pd100-di1.6-111-0.5.json](examples/input/owen-pd100-di1.6-111-0.5.json) | [md](examples/output/owen-pd100-di1.6-111-0.5.md) · [json](examples/output/owen-pd100-di1.6-111-0.5.json) | Датчик давления: описание серии + характеристики конкретной модификации |
| [manometer-dm02-from-task.json](examples/input/manometer-dm02-from-task.json) | [md](examples/output/manometer-dm02-from-task.md) · [json](examples/output/manometer-dm02-from-task.json) | Пример из ТЗ: мало данных — короткий честный текст вместо «воды», нет степени защиты |
| [examples/edge-cases/](examples/edge-cases) | — | Битый JSON, нет названия, пустые поля |

Результаты в `examples/output/` получены реальной моделью через OpenRouter командой выше.
Откуда взяты данные для примеров и что в них не так — в [examples/README.md](examples/README.md).

## Входные данные

JSON-объект. Обязательно только `name`, остальное — по возможности.

| Поле | Что это |
|---|---|
| `name` | Название / модель товара — **обязательно** |
| `manufacturer` | Производитель |
| `product_type` | Тип прибора («датчик давления общепромышленный») — помогает выбрать набор ключевых характеристик |
| `category` | Категория каталога |
| `specs` | Характеристики: `{"Диапазон измерений": "0–1,6 МПа"}` или `[{"name": ..., "value": ...}]` |
| `description` | Описание или фрагмент паспорта; HTML-теги очищаются |
| `identifiers` | Артикул, код товара — попадают в карточку, но **не передаются модели** |
| `source_url` | Ссылка на источник данных — для проверки человеком |

Поддерживаются синонимы полей: `vendor`/`brand` → `manufacturer`, `typePrefix` → `product_type`,
`params`/`characteristics` → `specs`, `url` → `source_url`.

## Результат

`examples/output/<имя>.md` — карточка для чтения и копирования, `examples/output/<имя>.json` — то же для систем:

```jsonc
{
  "status": "ok | needs_review",         // needs_review — есть что проверить человеку
  "seo_title": "...",
  "seo_description": "...",
  "key_specs": [{"name": "Степень защиты", "value": "данных нет", "source": "missing"}, ...],
  "search_queries": ["...", "..."],
  "commercial": {"summary": "...", "who_for": ["..."], "tasks": ["..."]},
  "missing_fields": ["Степень защиты"],  // чего нет во входных данных
  "issues": [{"level": "warning", "field": "description", "message": "..."}],
  "meta": {"model": "...", "attempts": 1, "response_format": "json_schema", "generated_at": "...",
           "prompt_version": 3, "judge_model": "...", "judge_status": "passed"}  // off | passed | issues | error
}
```

## Команды

```text
python -m kip_card generate FILE... [-o DIR] [-f md|json|both]   карточка по JSON ('-' — stdin)
python -m kip_card check FILE...                                  проверка входа без модели
python -m kip_card import-yml FEED --offer-id ID [-o DIR]         заготовка JSON из YML-фида
python -m kip_card audit-yml FEED                                 полнота данных в фиде
python -m kip_card serve [--host H] [--port P]                    веб-форма и API
```

Коды выхода: `0` — успех (карточка может иметь статус `needs_review`), `1` — ошибка входных данных,
`2` — ошибка настройки или модели.

## Веб-форма и API

`python -m kip_card serve` поднимает форму (ввод полями или JSON, кнопки «Копировать» у каждого блока,
вкладки Markdown/JSON) и HTTP API. Документация API — на `/docs`.

| Метод | Путь | Что делает |
|---|---|---|
| `POST` | `/api/generate` | JSON товара → `{"card": {...}, "markdown": "..."}` |
| `POST` | `/api/check` | Проверка входа без модели |
| `GET` | `/api/health` | Состояние и текущая модель |

Ошибки возвращаются с понятным текстом: `422` — некорректный вход (модель не вызывается),
`502` — модель недоступна или ответила ошибкой, `503` — сервис не настроен.

```bash
curl -X POST http://127.0.0.1:8000/api/generate \
     -H "Content-Type: application/json" \
     --data-binary @examples/input/testo-830-t1.json
```

Через этот эндпоинт сервис может вызывать сайт на 1С-Битрикс (например, кнопка «Сгенерировать
черновик» в карточке инфоблока) или обработка в 1С.

## Модель: облако или замкнутый контур

Клиент использует OpenAI-совместимый API, поэтому модель меняется в `.env` без правки кода.

| Где | `LLM_BASE_URL` | `LLM_MODEL` (пример) |
|---|---|---|
| OpenRouter (по умолчанию) | `https://openrouter.ai/api/v1` | `qwen/qwen3.8-27b` |
| Ollama | `http://localhost:11434/v1` | `qwen2.5:14b-instruct` |
| vLLM | `http://localhost:8000/v1` | `Qwen/Qwen2.5-14B-Instruct` |
| llama.cpp server | `http://localhost:8080/v1` | любое имя |

Генерацию и смысловую проверку выполняют разные модели — так их ошибки меньше совпадают.
Пара по умолчанию выбрана по сравнению трёх моделей на одних товарах
([DECISIONS, раздел 12](docs/DECISIONS.md#12-выбор-моделей-сравнение-на-одних-товарах)):

```bash
LLM_MODEL=qwen/qwen3.8-27b       # генерация: открытые веса, лучшие карточки в сравнении
JUDGE=on                         # on | off; примерно удваивает число запросов
JUDGE_MODEL=openai/gpt-5.4-mini  # проверка: другое семейство; пусто — та же модель, что LLM_MODEL
```

Для локальных серверов ключ не нужен. Если сервер не поддерживает строгую JSON-схему,
клиент сам переходит на JSON-режим, а затем на JSON по инструкции (`LLM_RESPONSE_FORMAT=auto`).
В замкнутом контуре генератор — локальная модель (Qwen3.8 27B в сжатом виде помещается на одну
видеокарту 24 ГБ), судья — тоже локальная, по возможности другого семейства.

## Docker

```bash
cp .env.example .env               # впишите ключ
docker compose up --build          # http://127.0.0.1:8000
```

Вместе с локальной моделью (Ollama в соседнем контейнере):

```bash
# в .env: LLM_BASE_URL=http://ollama:11434/v1  LLM_MODEL=qwen2.5:14b-instruct
docker compose --profile local up --build
docker compose exec ollama ollama pull qwen2.5:14b-instruct
```

## Импорт из YML-фида

Сайт выгружает каталог в формате YML. Импорт превращает товар фида во входной JSON:
название, тип, производитель, путь категории, ссылка, артикул. Цены, остатки и картинки
модели не передаются.

```bash
python -m kip_card audit-yml https://www.souz-pribor.ru/yml/any/feed.xml
python -m kip_card import-yml https://www.souz-pribor.ru/yml/any/feed.xml --offer-id 64423 -o data/
```

В текущем фиде нет ни описаний, ни технических характеристик (только «Артикул» и «Код товара»),
поэтому заготовку нужно дополнить данными со страницы товара или из паспорта. Без этого сервис
всё равно соберёт карточку, но с пометками «данных нет» и предупреждениями.

## Тесты

```bash
pip install -r requirements-dev.txt
pytest
```

136 тестов, без сети и без ключа (модель заменена заглушкой): разбор и проверка входа, фактчекер,
цикл исправления ответа, переход между режимами JSON, перевод ошибок API, веб-эндпоинты,
импорт фида (включая защиту от XML-атак), CLI. Отдельно — регрессия на настоящих ответах модели
из прогонов трёх моделей (`tests/fixtures/`): ошибки, найденные при вычитке, ловятся автоматически.

## Структура

```text
kip_card/
  config.py       настройки из окружения / .env
  validation.py   разбор JSON, нормализация, проверка полноты по типу прибора (без LLM)
  prompts.py      системный промпт, промпт товара, промпт исправления
  llm.py          OpenAI-совместимый клиент: json_schema → json_object → prompt, понятные ошибки
  factcheck.py    поиск чисел и технических кодов, которых нет во входных данных
  quality.py      заголовок, релевантность запросов, оценочные слова, «Кому подходит»
  judge.py        смысловая проверка вторым запросом к модели, проверка цитат кодом
  generator.py    сборка карточки: модель → проверки → исправление → карточка
  render.py       Markdown для контент-менеджера
  yml_import.py   импорт и аудит YML-фида
  cli.py, web.py  командная строка, FastAPI и форма (static/index.html)
examples/         входные данные и результаты
docs/DECISIONS.md промпты, решения, ограничения, улучшения
tests/            pytest
```

## Ограничения и улучшения

Кратко: единицы измерения не сверяются; смысловые искажения ловит модель-судья, но она тоже
модель и может что-то пропустить, поэтому результат — черновик для вычитки; длины
контролируются повторным запросом. Полный список и план развития — в
[docs/DECISIONS.md](docs/DECISIONS.md#ограничения).
