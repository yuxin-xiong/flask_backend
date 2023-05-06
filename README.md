# GSRTR (Grounded Situation Recognition with Transformers) Web
This is a flask project of Grounded Situation Recognition

v1.0 branch only return Verb

v1.1 branch return image object info eg:

```json
{
    "draw_url": "http://127.0.0.1:5000/static/output/zhuo_result.jpg",
    "image_url": "http://127.0.0.1:5000/static/input/zhuo.jpg",
    "res_info": [
        {
            "color": [
                232,
                126,
                253
            ],
            "noun": "woman",
            "role": "agent"
        },
        {
            "color": [
                130,
                234,
                198
            ],
            "noun": "floor",
            "role": "contact"
        },
        {
            "color": "null",
            "noun": "room",
            "role": "place"
        }
    ],
    "status": 1,
    "verb": "sitting"
}
```

v1.2 乱改

v1.3 差不多了 前一版本在v1.1上改的