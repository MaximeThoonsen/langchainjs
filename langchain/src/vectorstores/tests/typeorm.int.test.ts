import { expect, test } from "@jest/globals";
import { DataSourceOptions } from "typeorm";
import { OpenAIEmbeddings } from "../../embeddings/index.js";
import { TypeORMVectorStore } from "../typeorm.js";

test.skip("Test embeddings creation", async () => {
  const args = {
    postgresConnectionOptions: {
      type: "postgres",
      host: "localhost",
      port: 5432,
      username: "myuser",
      password: "!ChangeMe!",
      database: "api",
    } as DataSourceOptions,
  };

  const typeormVectorStore = await TypeORMVectorStore.fromDataSource(
    new OpenAIEmbeddings(),
    args
  );

  expect(typeormVectorStore).toBeDefined();

  await typeormVectorStore.initTable();

  const docHello = {
    pageContent: "hello",
    metadata: { a: 1 },
    sourceName: "hello.txt",
  };
  const docHi = { pageContent: "hi", metadata: { a: 1 }, sourceName: "hi.txt" };
  const docCat = {
    pageContent: "Cat drinks milk",
    metadata: { a: 2, sourceName: "cat.txt" },
  };

  await typeormVectorStore.addDocuments([
    docHello,
    docHi,
    docCat,
    { pageContent: "what's this", metadata: { a: 2, sourceName: "what.txt" } },
  ]);

  const results = await typeormVectorStore.similaritySearch("hello", 2);

  expect(results).toHaveLength(2);

  expect(results).toEqual([
    {
      ...docHello,
      id: "unknown:hello.txt:1-1",
      hash: "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d",
      sourceType: "unknown",
    },
    {
      ...docHi,
      id: "unknown:hi.txt:1-1",
      hash: "c22b5f9178342609428d6f51b2c5af4c0bde6a42",
      sourceType: "unknown",
    },
  ]);

  await typeormVectorStore.addDocuments([
    { ...docCat, pageContent: "Cat is drinking milk" },
  ]);

  const results2 = await typeormVectorStore.similaritySearch(
    "Cat drinks milk",
    1
  );

  expect(results2).toHaveLength(1);

  expect(results2[0].pageContent).toEqual("Cat is drinking milk");

  await typeormVectorStore.appDataSource.destroy();
});
