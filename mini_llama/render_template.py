from jinja2 import Template


def render_template(template, messages, tools=None):
    if tools is None:
        tools = []

    t = Template(template)

    return t.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=True,
    )