---
name: create-blog
description: Creates a blog post with proper structure and markdown features. Generates example content by default or actual content when explicitly requested.
---

# Content Generation Strategy

**Default Behavior**: When the user asks to create a blog post WITHOUT explicitly stating they want specific content, generate a blog post that demonstrates all the extended markdown features available on this site (admonitions, GitHub cards, spoilers, math, code blocks, tables, etc.).

**Explicit Content Request**: When the user explicitly states they want to create content about a specific topic, generate actual meaningful content for that topic.

## Ask the User

Before creating the blog post, ask the user:
1. What should the title be?
2. What category should it be in?
3. What tags should be applied?
4. (If not already clear) Should this be an example showcase of markdown features, or actual content about a specific topic?

---

When creating a blog post, always follow this structure:

## 1. Frontmatter (Required)

Every blog post must start with YAML frontmatter:

```yaml
---
title: Your Post Title
published: YYYY-MM-DD
description: 'A brief description shown on the index page'
image: './cover.jpg'
tags: [Tag1, Tag2]
category: 'Category Name'
draft: false
lang: 'en'
---
```

| Field | Required | Description |
|-------|----------|-------------|
| `title` | Yes | The title of the post |
| `published` | Yes | Publication date (YYYY-MM-DD) |
| `description` | Yes | Brief description for index page |
| `image` | No | Cover image (relative path, `/public` path, or URL) |
| `tags` | No | Array of tags |
| `category` | No | Single category |
| `draft` | No | Set `true` to hide from production |
| `lang` | No | Language code (e.g., 'en') |

## 2. File Location

Place blog posts in `src/content/posts/`:

```
src/content/posts/
├── simple-post.md                    # Simple post
└── post-with-assets/                 # Post with images
    ├── cover.jpg
    ├── diagram.png
    └── index.md
```

## 3. Extended Markdown Features

### Admonitions

Use callout boxes to highlight important information:

```markdown
:::note
Highlights information users should take into account.
:::

:::tip
Optional helpful information.
:::

:::important
Crucial information necessary for success.
:::

:::warning
Critical content demanding immediate attention.
:::

:::caution
Negative potential consequences of an action.
:::

:::note[CUSTOM TITLE]
Admonition with a custom title.
:::
```

GitHub syntax is also supported:

```markdown
> [!NOTE]
> GitHub-style callout.

> [!TIP]
> Another GitHub-style callout.
```

### GitHub Repository Cards

Embed dynamic GitHub repo cards:

```markdown
::github{repo="owner/repo-name"}
```

### Spoilers

Hide content with spoiler tags:

```markdown
The answer is :spoiler[hidden content here].
```

### Math Equations

Inline math: `$E = mc^2$`

Block math:
```markdown
$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$
```

### Code Blocks

Use fenced code blocks with language specification:

````markdown
```python
def hello():
    print("Hello, World!")
```

```c++
int main() {
    return 0;
}
```
````

### Tables

```markdown
| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |
```

# Content Generation Guidelines

## For Example/Demo Posts (Default)

Create a blog post that showcases ALL available markdown features:
- Use multiple types of admonitions (note, tip, important, warning, caution)
- Include at least one GitHub repository card
- Demonstrate spoiler text
- Show both inline and block math equations
- Include multiple code blocks with different languages
- Add a table
- Mix these features naturally throughout the post

The topic can be technical and interesting, but the primary goal is to demonstrate the markdown capabilities.

## For Explicit Content Requests

When the user explicitly asks for content about a specific topic:
1. Research or use knowledge about the topic
2. Create meaningful, valuable content
3. Still use appropriate markdown features where they enhance the content (code blocks for code examples, admonitions for important notes, etc.)
4. Focus on content quality over feature showcase

---

# Example: Markdown Feature Showcase Post

```markdown
---
title: Getting Started with CUDA
published: 2025-01-21
description: 'A beginner guide to GPU programming with CUDA - demonstrating markdown features'
image: './cuda-cover.png'
tags: [GPU, CUDA, Programming]
category: 'Tutorials'
draft: false
lang: 'en'
---

:::note
This post assumes basic C++ knowledge.
:::

# Introduction

GPU programming enables massive parallelism...

## Your First Kernel

Here's a simple vector addition kernel:

```c++
__global__ void vecAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

:::tip
Use prefix `d_` for device pointers and `h_` for host pointers.
:::

## Math Behind It

The complexity is $O(n)$ where:

$$
T_{total} = T_{transfer} + T_{compute}
$$

## Resources

::github{repo="NVIDIA/cuda-samples"}

:::warning
Always check CUDA errors in production code!
:::

The answer to the performance question is :spoiler[it depends on your memory bandwidth].
```
