using System.Reflection;
using YamlDotNet.Serialization;

namespace AliFsmnVad.Utils
{
    internal class PreloadHelper
    {
        public static T ReadYaml<T>(string yamlFilePath)
        {
            T? info = default(T);
            if (!string.IsNullOrEmpty(yamlFilePath) && yamlFilePath.IndexOf("/") < 0)
            {
                var assembly = Assembly.GetExecutingAssembly();
                var stream = assembly.GetManifestResourceStream(yamlFilePath) ??
                             throw new FileNotFoundException($"Embedded resource '{yamlFilePath}' not found.");
                using (var yamlReader = new StreamReader(stream))
                {
                    Deserializer yamlDeserializer = new Deserializer();
                    info = yamlDeserializer.Deserialize<T>(yamlReader);
                    yamlReader.Close();
                }
            }
            else if (File.Exists(yamlFilePath))
            {
                using (var yamlReader = File.OpenText(yamlFilePath))
                {
                    Deserializer yamlDeserializer = new Deserializer();
                    info = yamlDeserializer.Deserialize<T>(yamlReader);
                    yamlReader.Close();
                }
            }
#pragma warning disable CS8603 // 可能返回 null 引用。
            return info;
#pragma warning restore CS8603 // 可能返回 null 引用。
        }
    }
}
