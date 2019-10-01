# Blog

<ul>
  {% for post in site.posts %}
    <li>
      <a href="https://xilinx.github.io/finn/{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
