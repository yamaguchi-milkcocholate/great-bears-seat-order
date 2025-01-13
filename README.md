# great-bears-seat-order


## 手順
```shell
streamlit run app.py
```


## 備忘録
#### スプレッドシート連携
- GCPプロジェクトを作成
    - [great-bears-seet-order](https://console.cloud.google.com/welcome?inv=1&invt=Abmsdg&project=great-bears-seet-order)
- streamlitのprivate GS有効化
    - [ドキュメント](https://docs.streamlit.io/develop/tutorials/databases/private-gsheet)
    - APIを有効化: https://console.cloud.google.com/apis/api/sheets.googleapis.com/metrics?project=great-bears-seet-order&inv=1&invt=Abmsdg
    - サービスアカウントを作成: https://console.cloud.google.com/iam-admin/serviceaccounts?inv=1&invt=AbmshQ&project=great-bears-seet-order&supportedpurview=project
    
