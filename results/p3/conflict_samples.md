# Intra-Train Conflicting Normalized-Duplicate Samples
> Note: Raw code pairs shown side-by-side with metadata. Unjudged for manual review.

## Pair 1 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `primevul_train_train:10739_vuln`
- **Project:** `php-src`
- **Commit ID:** `523f230c831d7b33353203fa34aee4e92ac12bba`
- **CWE:** `None`
- **Tokens:** 13130

```c
php_stream *php_stream_url_wrap_http_ex(php_stream_wrapper *wrapper,
		const char *path, const char *mode, int options, char **opened_path,
		php_stream_context *context, int redirect_max, int flags STREAMS_DC TSRMLS_DC) /* {{{ */
{
	php_stream *stream = NULL;
	php_url *resource = NULL;
	int use_ssl;
	int use_proxy = 0;
	char *scratch = NULL;
	char *tmp = NULL;
	char *ua_str = NULL;
	zval **ua_zval = NULL, **tmpzval = NULL, *ssl_proxy_peer_name = NULL;
	int scratch_len = 0;
	int body = 0;
	char location[HTTP_HEADER_BLOCK_SIZE];
	zval *response_header = NULL;
	int reqok = 0;
	char *http_header_line = NULL;
	char tmp_line[128];
	size_t chunk_size = 0, file_size = 0;
	int eol_detect = 0;
	char *transport_string, *errstr = NULL;
	int transport_len, have_header = 0, request_fulluri = 0, ignore_errors = 0;
	char *protocol_version = NULL;
	int protocol_version_len = 3; /* Default: "1.0" */
	struct timeval timeout;
	char *user_headers = NULL;
	int header_init = ((flags & HTTP_WRAPPER_HEADER_INIT) != 0);
	int redirected = ((flags & HTTP_WRAPPER_REDIRECTED) != 0);
	int follow_location = 1;
	php_stream_filter *transfer_encoding = NULL;
	int response_code;

	tmp_line[0] = '\0';

	if (redirect_max < 1) {
		php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "Redirection limit reached, aborting");
		return NULL;
	}

	resource = php_url_parse(path);
	if (resource == NULL) {
		return NULL;
	}

	if (strncasecmp(resource->scheme, "http", sizeof("http")) && strncasecmp(resource->scheme, "https", sizeof("https"))) {
		if (!context ||
			php_stream_context_get_option(context, wrapper->wops->label, "proxy", &tmpzval) == FAILURE ||
			Z_TYPE_PP(tmpzval) != IS_STRING ||
			Z_STRLEN_PP(tmpzval) <= 0) {
			php_url_free(resource);
			return php_stream_open_wrapper_ex(path, mode, REPORT_ERRORS, NULL, context);
		}
		/* Called from a non-http wrapper with http proxying requested (i.e. ftp) */
		request_fulluri = 1;
		use_ssl = 0;
		use_proxy = 1;

		transport_len = Z_STRLEN_PP(tmpzval);
		transport_string = estrndup(Z_STRVAL_PP(tmpzval), Z_STRLEN_PP(tmpzval));
	} else {
		/* Normal http request (possibly with proxy) */

		if (strpbrk(mode, "awx+")) {
			php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "HTTP wrapper does not support writeable connections");
			php_url_free(resource);
			return NULL;
		}

		use_ssl = resource->scheme && (strlen(resource->scheme) > 4) && resource->scheme[4] == 's';
		/* choose default ports */
		if (use_ssl && resource->port == 0)
			resource->port = 443;
		else if (resource->port == 0)
			resource->port = 80;

		if (context &&
			php_stream_context_get_option(context, wrapper->wops->label, "proxy", &tmpzval) == SUCCESS &&
			Z_TYPE_PP(tmpzval) == IS_STRING &&
			Z_STRLEN_PP(tmpzval) > 0) {
			use_proxy = 1;
			transport_len = Z_STRLEN_PP(tmpzval);
			transport_string = estrndup(Z_STRVAL_PP(tmpzval), Z_STRLEN_PP(tmpzval));
		} else {
			transport_len = spprintf(&transport_string, 0, "%s://%s:%d", use_ssl ? "ssl" : "tcp", resource->host, resource->port);
		}
	}

	if (context && php_stream_context_get_option(context, wrapper->wops->label, "timeout", &tmpzval) == SUCCESS) {
		SEPARATE_ZVAL(tmpzval);
		convert_to_double_ex(tmpzval);
		timeout.tv_sec = (time_t) Z_DVAL_PP(tmpzval);
		timeout.tv_usec = (size_t) ((Z_DVAL_PP(tmpzval) - timeout.tv_sec) * 1000000);
	} else {
		timeout.tv_sec = FG(default_socket_timeout);
		timeout.tv_usec = 0;
	}

	stream = php_stream_xport_create(transport_string, transport_len, options,
			STREAM_XPORT_CLIENT | STREAM_XPORT_CONNECT,
			NULL, &timeout, context, &errstr, NULL);

	if (stream) {
		php_stream_set_option(stream, PHP_STREAM_OPTION_READ_TIMEOUT, 0, &timeout);
	}

	if (errstr) {
		php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "%s", errstr);
		efree(errstr);
		errstr = NULL;
	}

	efree(transport_string);

	if (stream && use_proxy && use_ssl) {
		smart_str header = {0};

		/* Set peer_name or name verification will try to use the proxy server name */
		if (!context || php_stream_context_get_option(context, "ssl", "peer_name", &tmpzval) == FAILURE) {
			MAKE_STD_ZVAL(ssl_proxy_peer_name);
			ZVAL_STRING(ssl_proxy_peer_name, resource->host, 1);
			php_stream_context_set_option(stream->context, "ssl", "peer_name", ssl_proxy_peer_name);
		}

		smart_str_appendl(&header, "CONNECT ", sizeof("CONNECT ")-1);
		smart_str_appends(&header, resource->host);
		smart_str_appendc(&header, ':');
		smart_str_append_unsigned(&header, resource->port);
		smart_str_appendl(&header, " HTTP/1.0\r\n", sizeof(" HTTP/1.0\r\n")-1);

	    /* check if we have Proxy-Authorization header */
		if (context && php_stream_context_get_option(context, "http", "header", &tmpzval) == SUCCESS) {
			char *s, *p;

			if (Z_TYPE_PP(tmpzval) == IS_ARRAY) {
				HashPosition pos;
				zval **tmpheader = NULL;

				for (zend_hash_internal_pointer_reset_ex(Z_ARRVAL_PP(tmpzval), &pos);
					SUCCESS == zend_hash_get_current_data_ex(Z_ARRVAL_PP(tmpzval), (void *)&tmpheader, &pos);
					zend_hash_move_forward_ex(Z_ARRVAL_PP(tmpzval), &pos)) {
					if (Z_TYPE_PP(tmpheader) == IS_STRING) {
						s = Z_STRVAL_PP(tmpheader);
						do {
							while (*s == ' ' || *s == '\t') s++;
							p = s;
							while (*p != 0 && *p != ':' && *p != '\r' && *p !='\n') p++;
							if (*p == ':') {
								p++;
								if (p - s == sizeof("Proxy-Authorization:") - 1 &&
								    zend_binary_strcasecmp(s, sizeof("Proxy-Authorization:") - 1,
								        "Proxy-Authorization:", sizeof("Proxy-Authorization:") - 1) == 0) {
									while (*p != 0 && *p != '\r' && *p !='\n') p++;
									smart_str_appendl(&header, s, p - s);
									smart_str_appendl(&header, "\r\n", sizeof("\r\n")-1);
									goto finish;
								} else {
									while (*p != 0 && *p != '\r' && *p !='\n') p++;
								}
							}
							s = p;
							while (*s == '\r' || *s == '\n') s++;
						} while (*s != 0);
					}
				}
			} else if (Z_TYPE_PP(tmpzval) == IS_STRING && Z_STRLEN_PP(tmpzval)) {
				s = Z_STRVAL_PP(tmpzval);
				do {
					while (*s == ' ' || *s == '\t') s++;
					p = s;
					while (*p != 0 && *p != ':' && *p != '\r' && *p !='\n') p++;
					if (*p == ':') {
						p++;
						if (p - s == sizeof("Proxy-Authorization:") - 1 &&
						    zend_binary_strcasecmp(s, sizeof("Proxy-Authorization:") - 1,
						        "Proxy-Authorization:", sizeof("Proxy-Authorization:") - 1) == 0) {
							while (*p != 0 && *p != '\r' && *p !='\n') p++;
							smart_str_appendl(&header, s, p - s);
							smart_str_appendl(&header, "\r\n", sizeof("\r\n")-1);
							goto finish;
						} else {
							while (*p != 0 && *p != '\r' && *p !='\n') p++;
						}
					}
					s = p;
					while (*s == '\r' || *s == '\n') s++;
				} while (*s != 0);
			}
		}
finish:
		smart_str_appendl(&header, "\r\n", sizeof("\r\n")-1);

		if (php_stream_write(stream, header.c, header.len) != header.len) {
			php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "Cannot connect to HTTPS server through proxy");
			php_stream_close(stream);
			stream = NULL;
		}
 	 	smart_str_free(&header);

 	 	if (stream) {
 	 		char header_line[HTTP_HEADER_BLOCK_SIZE];

			/* get response header */
			while (php_stream_gets(stream, header_line, HTTP_HEADER_BLOCK_SIZE-1) != NULL) {
				if (header_line[0] == '\n' ||
				    header_line[0] == '\r' ||
				    header_line[0] == '\0') {
				  break;
				}
			}
		}

		/* enable SSL transport layer */
		if (stream) {
			if (php_stream_xport_crypto_setup(stream, STREAM_CRYPTO_METHOD_SSLv23_CLIENT, NULL TSRMLS_CC) < 0 ||
			    php_stream_xport_crypto_enable(stream, 1 TSRMLS_CC) < 0) {
				php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "Cannot connect to HTTPS server through proxy");
				php_stream_close(stream);
				stream = NULL;
			}
		}
	}

	if (stream == NULL)
		goto out;

	/* avoid buffering issues while reading header */
	if (options & STREAM_WILL_CAST)
		chunk_size = php_stream_set_chunk_size(stream, 1);

	/* avoid problems with auto-detecting when reading the headers -> the headers
	 * are always in canonical \r\n format */
	eol_detect = stream->flags & (PHP_STREAM_FLAG_DETECT_EOL | PHP_STREAM_FLAG_EOL_MAC);
	stream->flags &= ~(PHP_STREAM_FLAG_DETECT_EOL | PHP_STREAM_FLAG_EOL_MAC);

	php_stream_context_set(stream, context);

	php_stream_notify_info(context, PHP_STREAM_NOTIFY_CONNECT, NULL, 0);

	if (header_init && context && php_stream_context_get_option(context, "http", "max_redirects", &tmpzval) == SUCCESS) {
		SEPARATE_ZVAL(tmpzval);
		convert_to_long_ex(tmpzval);
		redirect_max = Z_LVAL_PP(tmpzval);
	}

	if (context && php_stream_context_get_option(context, "http", "method", &tmpzval) == SUCCESS) {
		if (Z_TYPE_PP(tmpzval) == IS_STRING && Z_STRLEN_PP(tmpzval) > 0) {
			/* As per the RFC, automatically redirected requests MUST NOT use other methods than
			 * GET and HEAD unless it can be confirmed by the user */
			if (!redirected
				|| (Z_STRLEN_PP(tmpzval) == 3 && memcmp("GET", Z_STRVAL_PP(tmpzval), 3) == 0)
				|| (Z_STRLEN_PP(tmpzval) == 4 && memcmp("HEAD",Z_STRVAL_PP(tmpzval), 4) == 0)
			) {
				scratch_len = strlen(path) + 29 + Z_STRLEN_PP(tmpzval);
				scratch = emalloc(scratch_len);
				strlcpy(scratch, Z_STRVAL_PP(tmpzval), Z_STRLEN_PP(tmpzval) + 1);
				strncat(scratch, " ", 1);
			}
		}
	}

	if (context && php_stream_context_get_option(context, "http", "protocol_version", &tmpzval) == SUCCESS) {
		SEPARATE_ZVAL(tmpzval);
		convert_to_double_ex(tmpzval);
		protocol_version_len = spprintf(&protocol_version, 0, "%.1F", Z_DVAL_PP(tmpzval));
	}

	if (!scratch) {
		scratch_len = strlen(path) + 29 + protocol_version_len;
		scratch = emalloc(scratch_len);
		strncpy(scratch, "GET ", scratch_len);
	}

	/* Should we send the entire path in the request line, default to no. */
	if (!request_fulluri &&
		context &&
		php_stream_context_get_option(context, "http", "request_fulluri", &tmpzval) == SUCCESS) {
		zval ztmp = **tmpzval;

		zval_copy_ctor(&ztmp);
		convert_to_boolean(&ztmp);
		request_fulluri = Z_BVAL(ztmp) ? 1 : 0;
		zval_dtor(&ztmp);
	}

	if (request_fulluri) {
		/* Ask for everything */
		strcat(scratch, path);
	} else {
		/* Send the traditional /path/to/file?query_string */

		/* file */
		if (resource->path && *resource->path) {
			strlcat(scratch, resource->path, scratch_len);
		} else {
			strlcat(scratch, "/", scratch_len);
		}

		/* query string */
		if (resource->query) {
			strlcat(scratch, "?", scratch_len);
			strlcat(scratch, resource->query, scratch_len);
		}
	}

	/* protocol version we are speaking */
	if (protocol_version) {
		strlcat(scratch, " HTTP/", scratch_len);
		strlcat(scratch, protocol_version, scratch_len);
		strlcat(scratch, "\r\n", scratch_len);
	} else {
		strlcat(scratch, " HTTP/1.0\r\n", scratch_len);
	}

	/* send it */
	php_stream_write(stream, scratch, strlen(scratch));

	if (context && php_stream_context_get_option(context, "http", "header", &tmpzval) == SUCCESS) {
		tmp = NULL;

		if (Z_TYPE_PP(tmpzval) == IS_ARRAY) {
			HashPosition pos;
			zval **tmpheader = NULL;
			smart_str tmpstr = {0};

			for (zend_hash_internal_pointer_reset_ex(Z_ARRVAL_PP(tmpzval), &pos);
				SUCCESS == zend_hash_get_current_data_ex(Z_ARRVAL_PP(tmpzval), (void *)&tmpheader, &pos);
				zend_hash_move_forward_ex(Z_ARRVAL_PP(tmpzval), &pos)
			) {
				if (Z_TYPE_PP(tmpheader) == IS_STRING) {
					smart_str_appendl(&tmpstr, Z_STRVAL_PP(tmpheader), Z_STRLEN_PP(tmpheader));
					smart_str_appendl(&tmpstr, "\r\n", sizeof("\r\n") - 1);
				}
			}
			smart_str_0(&tmpstr);
			/* Remove newlines and spaces from start and end. there's at least one extra \r\n at the end that needs to go. */
			if (tmpstr.c) {
				tmp = php_trim(tmpstr.c, strlen(tmpstr.c), NULL, 0, NULL, 3 TSRMLS_CC);
				smart_str_free(&tmpstr);
			}
		}
		if (Z_TYPE_PP(tmpzval) == IS_STRING && Z_STRLEN_PP(tmpzval)) {
			/* Remove newlines and spaces from start and end php_trim will estrndup() */
			tmp = php_trim(Z_STRVAL_PP(tmpzval), Z_STRLEN_PP(tmpzval), NULL, 0, NULL, 3 TSRMLS_CC);
		}
		if (tmp && strlen(tmp) > 0) {
			char *s;

			user_headers = estrdup(tmp);

			/* Make lowercase for easy comparison against 'standard' headers */
			php_strtolower(tmp, strlen(tmp));

			if (!header_init) {
				/* strip POST headers on redirect */
				strip_header(user_headers, tmp, "content-length:");
				strip_header(user_headers, tmp, "content-type:");
			}

			if ((s = strstr(tmp, "user-agent:")) &&
			    (s == tmp || *(s-1) == '\r' || *(s-1) == '\n' ||
			                 *(s-1) == '\t' || *(s-1) == ' ')) {
				 have_header |= HTTP_HEADER_USER_AGENT;
			}
			if ((s = strstr(tmp, "host:")) &&
			    (s == tmp || *(s-1) == '\r' || *(s-1) == '\n' ||
			                 *(s-1) == '\t' || *(s-1) == ' ')) {
				 have_header |= HTTP_HEADER_HOST;
			}
			if ((s = strstr(tmp, "from:")) &&
			    (s == tmp || *(s-1) == '\r' || *(s-1) == '\n' ||
			                 *(s-1) == '\t' || *(s-1) == ' ')) {
				 have_header |= HTTP_HEADER_FROM;
				}
			if ((s = strstr(tmp, "authorization:")) &&
			    (s == tmp || *(s-1) == '\r' || *(s-1) == '\n' ||
			                 *(s-1) == '\t' || *(s-1) == ' ')) {
				 have_header |= HTTP_HEADER_AUTH;
			}
			if ((s = strstr(tmp, "content-length:")) &&
			    (s == tmp || *(s-1) == '\r' || *(s-1) == '\n' ||
			                 *(s-1) == '\t' || *(s-1) == ' ')) {
				 have_header |= HTTP_HEADER_CONTENT_LENGTH;
			}
			if ((s = strstr(tmp, "content-type:")) &&
			    (s == tmp || *(s-1) == '\r' || *(s-1) == '\n' ||
			                 *(s-1) == '\t' || *(s-1) == ' ')) {
				 have_header |= HTTP_HEADER_TYPE;
			}
			if ((s = strstr(tmp, "connection:")) &&
			    (s == tmp || *(s-1) == '\r' || *(s-1) == '\n' ||
			                 *(s-1) == '\t' || *(s-1) == ' ')) {
				 have_header |= HTTP_HEADER_CONNECTION;
			}
			/* remove Proxy-Authorization header */
			if (use_proxy && use_ssl && (s = strstr(tmp, "proxy-authorization:")) &&
			    (s == tmp || *(s-1) == '\r' || *(s-1) == '\n' ||
			                 *(s-1) == '\t' || *(s-1) == ' ')) {
				char *p = s + sizeof("proxy-authorization:") - 1;

				while (s > tmp && (*(s-1) == ' ' || *(s-1) == '\t')) s--;
				while (*p != 0 && *p != '\r' && *p != '\n') p++;
				while (*p == '\r' || *p == '\n') p++;
				if (*p == 0) {
					if (s == tmp) {
						efree(user_headers);
						user_headers = NULL;
					} else {
						while (s > tmp && (*(s-1) == '\r' || *(s-1) == '\n')) s--;
						user_headers[s - tmp] = 0;
					}
				} else {
					memmove(user_headers + (s - tmp), user_headers + (p - tmp), strlen(p) + 1);
				}
			}

		}
		if (tmp) {
			efree(tmp);
		}
	}

	/* auth header if it was specified */
	if (((have_header & HTTP_HEADER_AUTH) == 0) && resource->user) {
		/* decode the strings first */
		php_url_decode(resource->user, strlen(resource->user));

		/* scratch is large enough, since it was made large enough for the whole URL */
		strcpy(scratch, resource->user);
		strcat(scratch, ":");

		/* Note: password is optional! */
		if (resource->pass) {
			php_url_decode(resource->pass, strlen(resource->pass));
			strcat(scratch, resource->pass);
		}

		tmp = (char*)php_base64_encode((unsigned char*)scratch, strlen(scratch), NULL);

		if (snprintf(scratch, scratch_len, "Authorization: Basic %s\r\n", tmp) > 0) {
			php_stream_write(stream, scratch, strlen(scratch));
			php_stream_notify_info(context, PHP_STREAM_NOTIFY_AUTH_REQUIRED, NULL, 0);
		}

		efree(tmp);
		tmp = NULL;
	}

	/* if the user has configured who they are, send a From: line */
	if (((have_header & HTTP_HEADER_FROM) == 0) && FG(from_address)) {
		if (snprintf(scratch, scratch_len, "From: %s\r\n", FG(from_address)) > 0)
			php_stream_write(stream, scratch, strlen(scratch));
	}

	/* Send Host: header so name-based virtual hosts work */
	if ((have_header & HTTP_HEADER_HOST) == 0) {
		if ((use_ssl && resource->port != 443 && resource->port != 0) ||
			(!use_ssl && resource->port != 80 && resource->port != 0)) {
			if (snprintf(scratch, scratch_len, "Host: %s:%i\r\n", resource->host, resource->port) > 0)
				php_stream_write(stream, scratch, strlen(scratch));
		} else {
			if (snprintf(scratch, scratch_len, "Host: %s\r\n", resource->host) > 0) {
				php_stream_write(stream, scratch, strlen(scratch));
			}
		}
	}

	/* Send a Connection: close header to avoid hanging when the server
	 * interprets the RFC literally and establishes a keep-alive connection,
	 * unless the user specifically requests something else by specifying a
	 * Connection header in the context options. Send that header even for
	 * HTTP/1.0 to avoid issues when the server respond with a HTTP/1.1
	 * keep-alive response, which is the preferred response type. */
	if ((have_header & HTTP_HEADER_CONNECTION) == 0) {
		php_stream_write_string(stream, "Connection: close\r\n");
	}

	if (context &&
	    php_stream_context_get_option(context, "http", "user_agent", &ua_zval) == SUCCESS &&
		Z_TYPE_PP(ua_zval) == IS_STRING) {
		ua_str = Z_STRVAL_PP(ua_zval);
	} else if (FG(user_agent)) {
		ua_str = FG(user_agent);
	}

	if (((have_header & HTTP_HEADER_USER_AGENT) == 0) && ua_str) {
#define _UA_HEADER "User-Agent: %s\r\n"
		char *ua;
		size_t ua_len;

		ua_len = sizeof(_UA_HEADER) + strlen(ua_str);

		/* ensure the header is only sent if user_agent is not blank */
		if (ua_len > sizeof(_UA_HEADER)) {
			ua = emalloc(ua_len + 1);
			if ((ua_len = slprintf(ua, ua_len, _UA_HEADER, ua_str)) > 0) {
				ua[ua_len] = 0;
				php_stream_write(stream, ua, ua_len);
			} else {
				php_error_docref(NULL TSRMLS_CC, E_WARNING, "Cannot construct User-agent header");
			}

			if (ua) {
				efree(ua);
			}
		}
	}

	if (user_headers) {
		/* A bit weird, but some servers require that Content-Length be sent prior to Content-Type for POST
		 * see bug #44603 for details. Since Content-Type maybe part of user's headers we need to do this check first.
		 */
		if (
				header_init &&
				context &&
				!(have_header & HTTP_HEADER_CONTENT_LENGTH) &&
				php_stream_context_get_option(context, "http", "content", &tmpzval) == SUCCESS &&
				Z_TYPE_PP(tmpzval) == IS_STRING && Z_STRLEN_PP(tmpzval) > 0
		) {
			scratch_len = slprintf(scratch, scratch_len, "Content-Length: %d\r\n", Z_STRLEN_PP(tmpzval));
			php_stream_write(stream, scratch, scratch_len);
			have_header |= HTTP_HEADER_CONTENT_LENGTH;
		}

		php_stream_write(stream, user_headers, strlen(user_headers));
		php_stream_write(stream, "\r\n", sizeof("\r\n")-1);
		efree(user_headers);
	}

	/* Request content, such as for POST requests */
	if (header_init && context &&
		php_stream_context_get_option(context, "http", "content", &tmpzval) == SUCCESS &&
		Z_TYPE_PP(tmpzval) == IS_STRING && Z_STRLEN_PP(tmpzval) > 0) {
		if (!(have_header & HTTP_HEADER_CONTENT_LENGTH)) {
			scratch_len = slprintf(scratch, scratch_len, "Content-Length: %d\r\n", Z_STRLEN_PP(tmpzval));
			php_stream_write(stream, scratch, scratch_len);
		}
		if (!(have_header & HTTP_HEADER_TYPE)) {
			php_stream_write(stream, "Content-Type: application/x-www-form-urlencoded\r\n",
				sizeof("Content-Type: application/x-www-form-urlencoded\r\n") - 1);
			php_error_docref(NULL TSRMLS_CC, E_NOTICE, "Content-type not specified assuming application/x-www-form-urlencoded");
		}
		php_stream_write(stream, "\r\n", sizeof("\r\n")-1);
		php_stream_write(stream, Z_STRVAL_PP(tmpzval), Z_STRLEN_PP(tmpzval));
	} else {
		php_stream_write(stream, "\r\n", sizeof("\r\n")-1);
	}

	location[0] = '\0';

	if (!EG(active_symbol_table)) {
		zend_rebuild_symbol_table(TSRMLS_C);
	}

	if (header_init) {
		zval *ztmp;
		MAKE_STD_ZVAL(ztmp);
		array_init(ztmp);
		ZEND_SET_SYMBOL(EG(active_symbol_table), "http_response_header", ztmp);
	}

	{
		zval **rh;
		if(zend_hash_find(EG(active_symbol_table), "http_response_header", sizeof("http_response_header"), (void **) &rh) != SUCCESS || Z_TYPE_PP(rh) != IS_ARRAY) {
			php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "HTTP request failed, http_response_header overwritten");
			goto out;
		}
		response_header = *rh;
		Z_ADDREF_P(response_header);
	}

	if (!php_stream_eof(stream)) {
		size_t tmp_line_len;
		/* get response header */

		if (php_stream_get_line(stream, tmp_line, sizeof(tmp_line) - 1, &tmp_line_len) != NULL) {
			zval *http_response;

			if (tmp_line_len > 9) {
				response_code = atoi(tmp_line + 9);
			} else {
				response_code = 0;
			}
			if (context && SUCCESS==php_stream_context_get_option(context, "http", "ignore_errors", &tmpzval)) {
				ignore_errors = zend_is_true(*tmpzval);
			}
			/* when we request only the header, don't fail even on error codes */
			if ((options & STREAM_ONLY_GET_HEADERS) || ignore_errors) {
				reqok = 1;
			}

			/* status codes of 1xx are "informational", and will be followed by a real response
			 * e.g "100 Continue". RFC 7231 states that unexpected 1xx status MUST be parsed,
			 * and MAY be ignored. As such, we need to skip ahead to the "real" status*/
			if (response_code >= 100 && response_code < 200) {
				/* consume lines until we find a line starting 'HTTP/1' */
				while (
					!php_stream_eof(stream)
					&& php_stream_get_line(stream, tmp_line, sizeof(tmp_line) - 1, &tmp_line_len) != NULL
					&& ( tmp_line_len < sizeof("HTTP/1") - 1 || strncasecmp(tmp_line, "HTTP/1", sizeof("HTTP/1") - 1) )
				);

				if (tmp_line_len > 9) {
					response_code = atoi(tmp_line + 9);
				} else {
					response_code = 0;
				}
			}
			/* all status codes in the 2xx range are defined by the specification as successful;
			 * all status codes in the 3xx range are for redirection, and so also should never
			 * fail */
			if (response_code >= 200 && response_code < 400) {
				reqok = 1;
			} else {
				switch(response_code) {
					case 403:
						php_stream_notify_error(context, PHP_STREAM_NOTIFY_AUTH_RESULT,
								tmp_line, response_code);
						break;
					default:
						/* safety net in the event tmp_line == NULL */
						if (!tmp_line_len) {
							tmp_line[0] = '\0';
						}
						php_stream_notify_error(context, PHP_STREAM_NOTIFY_FAILURE,
								tmp_line, response_code);
				}
			}
			if (tmp_line[tmp_line_len - 1] == '\n') {
				--tmp_line_len;
				if (tmp_line[tmp_line_len - 1] == '\r') {
					--tmp_line_len;
				}
			}
			MAKE_STD_ZVAL(http_response);
			ZVAL_STRINGL(http_response, tmp_line, tmp_line_len, 1);
			zend_hash_next_index_insert(Z_ARRVAL_P(response_header), &http_response, sizeof(zval *), NULL);
		}
	} else {
		php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "HTTP request failed, unexpected end of socket!");
		goto out;
	}

	/* read past HTTP headers */

	http_header_line = emalloc(HTTP_HEADER_BLOCK_SIZE);

	while (!body && !php_stream_eof(stream)) {
		size_t http_header_line_length;
		if (php_stream_get_line(stream, http_header_line, HTTP_HEADER_BLOCK_SIZE, &http_header_line_length) && *http_header_line != '\n' && *http_header_line != '\r') {
			char *e = http_header_line + http_header_line_length - 1;
			if (*e != '\n') {
				do { /* partial header */
					if (php_stream_get_line(stream, http_header_line, HTTP_HEADER_BLOCK_SIZE, &http_header_line_length) == NULL) {
						php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "Failed to read HTTP headers");
						goto out;
					}
					e = http_header_line + http_header_line_length - 1;
				} while (*e != '\n');
				continue;
			}
			while (*e == '\n' || *e == '\r') {
				e--;
			}
			http_header_line_length = e - http_header_line + 1;
			http_header_line[http_header_line_length] = '\0';

			if (!strncasecmp(http_header_line, "Location: ", 10)) {
				if (context && php_stream_context_get_option(context, "http", "follow_location", &tmpzval) == SUCCESS) {
					SEPARATE_ZVAL(tmpzval);
					convert_to_long_ex(tmpzval);
					follow_location = Z_LVAL_PP(tmpzval);
				} else if (!(response_code >= 300 && response_code < 304 || 307 == response_code || 308 == response_code)) {
					/* we shouldn't redirect automatically
					if follow_location isn't set and response_code not in (300, 301, 302, 303 and 307)
					see http://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html#sec10.3.1
					RFC 7238 defines 308: http://tools.ietf.org/html/rfc7238 */
					follow_location = 0;
				}
				strlcpy(location, http_header_line + 10, sizeof(location));
			} else if (!strncasecmp(http_header_line, "Content-Type: ", 14)) {
				php_stream_notify_info(context, PHP_STREAM_NOTIFY_MIME_TYPE_IS, http_header_line + 14, 0);
			} else if (!strncasecmp(http_header_line, "Content-Length: ", 16)) {
				file_size = atoi(http_header_line + 16);
				php_stream_notify_file_size(context, file_size, http_header_line, 0);
			} else if (!strncasecmp(http_header_line, "Transfer-Encoding: chunked", sizeof("Transfer-Encoding: chunked"))) {

				/* create filter to decode response body */
				if (!(options & STREAM_ONLY_GET_HEADERS)) {
					long decode = 1;

					if (context && php_stream_context_get_option(context, "http", "auto_decode", &tmpzval) == SUCCESS) {
						SEPARATE_ZVAL(tmpzval);
						convert_to_boolean(*tmpzval);
						decode = Z_LVAL_PP(tmpzval);
					}
					if (decode) {
						transfer_encoding = php_stream_filter_create("dechunk", NULL, php_stream_is_persistent(stream) TSRMLS_CC);
						if (transfer_encoding) {
							/* don't store transfer-encodeing header */
							continue;
						}
					}
				}
			}

			if (http_header_line[0] == '\0') {
				body = 1;
			} else {
				zval *http_header;

				MAKE_STD_ZVAL(http_header);

				ZVAL_STRINGL(http_header, http_header_line, http_header_line_length, 1);

				zend_hash_next_index_insert(Z_ARRVAL_P(response_header), &http_header, sizeof(zval *), NULL);
			}
		} else {
			break;
		}
	}

	if (!reqok || (location[0] != '\0' && follow_location)) {
		if (!follow_location || (((options & STREAM_ONLY_GET_HEADERS) || ignore_errors) && redirect_max <= 1)) {
			goto out;
		}

		if (location[0] != '\0')
			php_stream_notify_info(context, PHP_STREAM_NOTIFY_REDIRECTED, location, 0);

		php_stream_close(stream);
		stream = NULL;

		if (location[0] != '\0') {

			char new_path[HTTP_HEADER_BLOCK_SIZE];
			char loc_path[HTTP_HEADER_BLOCK_SIZE];

			*new_path='\0';
			if (strlen(location)<8 || (strncasecmp(location, "http://", sizeof("http://")-1) &&
							strncasecmp(location, "https://", sizeof("https://")-1) &&
							strncasecmp(location, "ftp://", sizeof("ftp://")-1) &&
							strncasecmp(location, "ftps://", sizeof("ftps://")-1)))
			{
				if (*location != '/') {
					if (*(location+1) != '\0' && resource->path) {
						char *s = strrchr(resource->path, '/');
						if (!s) {
							s = resource->path;
							if (!s[0]) {
								efree(s);
								s = resource->path = estrdup("/");
							} else {
								*s = '/';
							}
						}
						s[1] = '\0';
						if (resource->path && *(resource->path) == '/' && *(resource->path + 1) == '\0') {
							snprintf(loc_path, sizeof(loc_path) - 1, "%s%s", resource->path, location);
						} else {
							snprintf(loc_path, sizeof(loc_path) - 1, "%s/%s", resource->path, location);
						}
					} else {
						snprintf(loc_path, sizeof(loc_path) - 1, "/%s", location);
					}
				} else {
					strlcpy(loc_path, location, sizeof(loc_path));
				}
				if ((use_ssl && resource->port != 443) || (!use_ssl && resource->port != 80)) {
					snprintf(new_path, sizeof(new_path) - 1, "%s://%s:%d%s", resource->scheme, resource->host, resource->port, loc_path);
				} else {
					snprintf(new_path, sizeof(new_path) - 1, "%s://%s%s", resource->scheme, resource->host, loc_path);
				}
			} else {
				strlcpy(new_path, location, sizeof(new_path));
			}

			php_url_free(resource);
			/* check for invalid redirection URLs */
			if ((resource = php_url_parse(new_path)) == NULL) {
				php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "Invalid redirect URL! %s", new_path);
				goto out;
			}

#define CHECK_FOR_CNTRL_CHARS(val) { \
	if (val) { \
		unsigned char *s, *e; \
		int l; \
		l = php_url_decode(val, strlen(val)); \
		s = (unsigned char*)val; e = s + l; \
		while (s < e) { \
			if (iscntrl(*s)) { \
				php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "Invalid redirect URL! %s", new_path); \
				goto out; \
			} \
			s++; \
		} \
	} \
}
			/* check for control characters in login, password & path */
			if (strncasecmp(new_path, "http://", sizeof("http://") - 1) || strncasecmp(new_path, "https://", sizeof("https://") - 1)) {
				CHECK_FOR_CNTRL_CHARS(resource->user)
				CHECK_FOR_CNTRL_CHARS(resource->pass)
				CHECK_FOR_CNTRL_CHARS(resource->path)
			}
			stream = php_stream_url_wrap_http_ex(wrapper, new_path, mode, options, opened_path, context, --redirect_max, HTTP_WRAPPER_REDIRECTED STREAMS_CC TSRMLS_CC);
		} else {
			php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "HTTP request failed! %s", tmp_line);
		}
	}
out:
	if (protocol_version) {
		efree(protocol_version);
	}

	if (http_header_line) {
		efree(http_header_line);
	}

	if (scratch) {
		efree(scratch);
	}

	if (resource) {
		php_url_free(resource);
	}

	if (stream) {
		if (header_init) {
			stream->wrapperdata = response_header;
		} else {
			if(response_header) {
				Z_DELREF_P(response_header);
			}
		}
		php_stream_notify_progress_init(context, 0, file_size);

		/* Restore original chunk size now that we're done with headers */
		if (options & STREAM_WILL_CAST)
			php_stream_set_chunk_size(stream, chunk_size);

		/* restore the users auto-detect-line-endings setting */
		stream->flags |= eol_detect;

		/* as far as streams are concerned, we are now at the start of
		 * the stream */
		stream->position = 0;

		/* restore mode */
		strlcpy(stream->mode, mode, sizeof(stream->mode));

		if (transfer_encoding) {
			php_stream_filter_append(&stream->readfilters, transfer_encoding);
		}
	} else {
		if(response_header) {
			Z_DELREF_P(response_header);
		}
		if (transfer_encoding) {
			php_stream_filter_free(transfer_encoding TSRMLS_CC);
		}
	}

	return stream;
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `primevul_train_train:10739_fixed`
- **Project:** `php-src`
- **Commit ID:** `523f230c831d7b33353203fa34aee4e92ac12bba`
- **Tokens:** 13158

```c
php_stream *php_stream_url_wrap_http_ex(php_stream_wrapper *wrapper,
		const char *path, const char *mode, int options, char **opened_path,
		php_stream_context *context, int redirect_max, int flags STREAMS_DC TSRMLS_DC) /* {{{ */
{
	php_stream *stream = NULL;
	php_url *resource = NULL;
	int use_ssl;
	int use_proxy = 0;
	char *scratch = NULL;
	char *tmp = NULL;
	char *ua_str = NULL;
	zval **ua_zval = NULL, **tmpzval = NULL, *ssl_proxy_peer_name = NULL;
	int scratch_len = 0;
	int body = 0;
	char location[HTTP_HEADER_BLOCK_SIZE];
	zval *response_header = NULL;
	int reqok = 0;
	char *http_header_line = NULL;
	char tmp_line[128];
	size_t chunk_size = 0, file_size = 0;
	int eol_detect = 0;
	char *transport_string, *errstr = NULL;
	int transport_len, have_header = 0, request_fulluri = 0, ignore_errors = 0;
	char *protocol_version = NULL;
	int protocol_version_len = 3; /* Default: "1.0" */
	struct timeval timeout;
	char *user_headers = NULL;
	int header_init = ((flags & HTTP_WRAPPER_HEADER_INIT) != 0);
	int redirected = ((flags & HTTP_WRAPPER_REDIRECTED) != 0);
	int follow_location = 1;
	php_stream_filter *transfer_encoding = NULL;
	int response_code;

	tmp_line[0] = '\0';

	if (redirect_max < 1) {
		php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "Redirection limit reached, aborting");
		return NULL;
	}

	resource = php_url_parse(path);
	if (resource == NULL) {
		return NULL;
	}

	if (strncasecmp(resource->scheme, "http", sizeof("http")) && strncasecmp(resource->scheme, "https", sizeof("https"))) {
		if (!context ||
			php_stream_context_get_option(context, wrapper->wops->label, "proxy", &tmpzval) == FAILURE ||
			Z_TYPE_PP(tmpzval) != IS_STRING ||
			Z_STRLEN_PP(tmpzval) <= 0) {
			php_url_free(resource);
			return php_stream_open_wrapper_ex(path, mode, REPORT_ERRORS, NULL, context);
		}
		/* Called from a non-http wrapper with http proxying requested (i.e. ftp) */
		request_fulluri = 1;
		use_ssl = 0;
		use_proxy = 1;

		transport_len = Z_STRLEN_PP(tmpzval);
		transport_string = estrndup(Z_STRVAL_PP(tmpzval), Z_STRLEN_PP(tmpzval));
	} else {
		/* Normal http request (possibly with proxy) */

		if (strpbrk(mode, "awx+")) {
			php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "HTTP wrapper does not support writeable connections");
			php_url_free(resource);
			return NULL;
		}

		use_ssl = resource->scheme && (strlen(resource->scheme) > 4) && resource->scheme[4] == 's';
		/* choose default ports */
		if (use_ssl && resource->port == 0)
			resource->port = 443;
		else if (resource->port == 0)
			resource->port = 80;

		if (context &&
			php_stream_context_get_option(context, wrapper->wops->label, "proxy", &tmpzval) == SUCCESS &&
			Z_TYPE_PP(tmpzval) == IS_STRING &&
			Z_STRLEN_PP(tmpzval) > 0) {
			use_proxy = 1;
			transport_len = Z_STRLEN_PP(tmpzval);
			transport_string = estrndup(Z_STRVAL_PP(tmpzval), Z_STRLEN_PP(tmpzval));
		} else {
			transport_len = spprintf(&transport_string, 0, "%s://%s:%d", use_ssl ? "ssl" : "tcp", resource->host, resource->port);
		}
	}

	if (context && php_stream_context_get_option(context, wrapper->wops->label, "timeout", &tmpzval) == SUCCESS) {
		SEPARATE_ZVAL(tmpzval);
		convert_to_double_ex(tmpzval);
		timeout.tv_sec = (time_t) Z_DVAL_PP(tmpzval);
		timeout.tv_usec = (size_t) ((Z_DVAL_PP(tmpzval) - timeout.tv_sec) * 1000000);
	} else {
		timeout.tv_sec = FG(default_socket_timeout);
		timeout.tv_usec = 0;
	}

	stream = php_stream_xport_create(transport_string, transport_len, options,
			STREAM_XPORT_CLIENT | STREAM_XPORT_CONNECT,
			NULL, &timeout, context, &errstr, NULL);

	if (stream) {
		php_stream_set_option(stream, PHP_STREAM_OPTION_READ_TIMEOUT, 0, &timeout);
	}

	if (errstr) {
		php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "%s", errstr);
		efree(errstr);
		errstr = NULL;
	}

	efree(transport_string);

	if (stream && use_proxy && use_ssl) {
		smart_str header = {0};

		/* Set peer_name or name verification will try to use the proxy server name */
		if (!context || php_stream_context_get_option(context, "ssl", "peer_name", &tmpzval) == FAILURE) {
			MAKE_STD_ZVAL(ssl_proxy_peer_name);
			ZVAL_STRING(ssl_proxy_peer_name, resource->host, 1);
			php_stream_context_set_option(stream->context, "ssl", "peer_name", ssl_proxy_peer_name);
		}

		smart_str_appendl(&header, "CONNECT ", sizeof("CONNECT ")-1);
		smart_str_appends(&header, resource->host);
		smart_str_appendc(&header, ':');
		smart_str_append_unsigned(&header, resource->port);
		smart_str_appendl(&header, " HTTP/1.0\r\n", sizeof(" HTTP/1.0\r\n")-1);

	    /* check if we have Proxy-Authorization header */
		if (context && php_stream_context_get_option(context, "http", "header", &tmpzval) == SUCCESS) {
			char *s, *p;

			if (Z_TYPE_PP(tmpzval) == IS_ARRAY) {
				HashPosition pos;
				zval **tmpheader = NULL;

				for (zend_hash_internal_pointer_reset_ex(Z_ARRVAL_PP(tmpzval), &pos);
					SUCCESS == zend_hash_get_current_data_ex(Z_ARRVAL_PP(tmpzval), (void *)&tmpheader, &pos);
					zend_hash_move_forward_ex(Z_ARRVAL_PP(tmpzval), &pos)) {
					if (Z_TYPE_PP(tmpheader) == IS_STRING) {
						s = Z_STRVAL_PP(tmpheader);
						do {
							while (*s == ' ' || *s == '\t') s++;
							p = s;
							while (*p != 0 && *p != ':' && *p != '\r' && *p !='\n') p++;
							if (*p == ':') {
								p++;
								if (p - s == sizeof("Proxy-Authorization:") - 1 &&
								    zend_binary_strcasecmp(s, sizeof("Proxy-Authorization:") - 1,
								        "Proxy-Authorization:", sizeof("Proxy-Authorization:") - 1) == 0) {
									while (*p != 0 && *p != '\r' && *p !='\n') p++;
									smart_str_appendl(&header, s, p - s);
									smart_str_appendl(&header, "\r\n", sizeof("\r\n")-1);
									goto finish;
								} else {
									while (*p != 0 && *p != '\r' && *p !='\n') p++;
								}
							}
							s = p;
							while (*s == '\r' || *s == '\n') s++;
						} while (*s != 0);
					}
				}
			} else if (Z_TYPE_PP(tmpzval) == IS_STRING && Z_STRLEN_PP(tmpzval)) {
				s = Z_STRVAL_PP(tmpzval);
				do {
					while (*s == ' ' || *s == '\t') s++;
					p = s;
					while (*p != 0 && *p != ':' && *p != '\r' && *p !='\n') p++;
					if (*p == ':') {
						p++;
						if (p - s == sizeof("Proxy-Authorization:") - 1 &&
						    zend_binary_strcasecmp(s, sizeof("Proxy-Authorization:") - 1,
						        "Proxy-Authorization:", sizeof("Proxy-Authorization:") - 1) == 0) {
							while (*p != 0 && *p != '\r' && *p !='\n') p++;
							smart_str_appendl(&header, s, p - s);
							smart_str_appendl(&header, "\r\n", sizeof("\r\n")-1);
							goto finish;
						} else {
							while (*p != 0 && *p != '\r' && *p !='\n') p++;
						}
					}
					s = p;
					while (*s == '\r' || *s == '\n') s++;
				} while (*s != 0);
			}
		}
finish:
		smart_str_appendl(&header, "\r\n", sizeof("\r\n")-1);

		if (php_stream_write(stream, header.c, header.len) != header.len) {
			php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "Cannot connect to HTTPS server through proxy");
			php_stream_close(stream);
			stream = NULL;
		}
 	 	smart_str_free(&header);

 	 	if (stream) {
 	 		char header_line[HTTP_HEADER_BLOCK_SIZE];

			/* get response header */
			while (php_stream_gets(stream, header_line, HTTP_HEADER_BLOCK_SIZE-1) != NULL) {
				if (header_line[0] == '\n' ||
				    header_line[0] == '\r' ||
				    header_line[0] == '\0') {
				  break;
				}
			}
		}

		/* enable SSL transport layer */
		if (stream) {
			if (php_stream_xport_crypto_setup(stream, STREAM_CRYPTO_METHOD_SSLv23_CLIENT, NULL TSRMLS_CC) < 0 ||
			    php_stream_xport_crypto_enable(stream, 1 TSRMLS_CC) < 0) {
				php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "Cannot connect to HTTPS server through proxy");
				php_stream_close(stream);
				stream = NULL;
			}
		}
	}

	if (stream == NULL)
		goto out;

	/* avoid buffering issues while reading header */
	if (options & STREAM_WILL_CAST)
		chunk_size = php_stream_set_chunk_size(stream, 1);

	/* avoid problems with auto-detecting when reading the headers -> the headers
	 * are always in canonical \r\n format */
	eol_detect = stream->flags & (PHP_STREAM_FLAG_DETECT_EOL | PHP_STREAM_FLAG_EOL_MAC);
	stream->flags &= ~(PHP_STREAM_FLAG_DETECT_EOL | PHP_STREAM_FLAG_EOL_MAC);

	php_stream_context_set(stream, context);

	php_stream_notify_info(context, PHP_STREAM_NOTIFY_CONNECT, NULL, 0);

	if (header_init && context && php_stream_context_get_option(context, "http", "max_redirects", &tmpzval) == SUCCESS) {
		SEPARATE_ZVAL(tmpzval);
		convert_to_long_ex(tmpzval);
		redirect_max = Z_LVAL_PP(tmpzval);
	}

	if (context && php_stream_context_get_option(context, "http", "method", &tmpzval) == SUCCESS) {
		if (Z_TYPE_PP(tmpzval) == IS_STRING && Z_STRLEN_PP(tmpzval) > 0) {
			/* As per the RFC, automatically redirected requests MUST NOT use other methods than
			 * GET and HEAD unless it can be confirmed by the user */
			if (!redirected
				|| (Z_STRLEN_PP(tmpzval) == 3 && memcmp("GET", Z_STRVAL_PP(tmpzval), 3) == 0)
				|| (Z_STRLEN_PP(tmpzval) == 4 && memcmp("HEAD",Z_STRVAL_PP(tmpzval), 4) == 0)
			) {
				scratch_len = strlen(path) + 29 + Z_STRLEN_PP(tmpzval);
				scratch = emalloc(scratch_len);
				strlcpy(scratch, Z_STRVAL_PP(tmpzval), Z_STRLEN_PP(tmpzval) + 1);
				strncat(scratch, " ", 1);
			}
		}
	}

	if (context && php_stream_context_get_option(context, "http", "protocol_version", &tmpzval) == SUCCESS) {
		SEPARATE_ZVAL(tmpzval);
		convert_to_double_ex(tmpzval);
		protocol_version_len = spprintf(&protocol_version, 0, "%.1F", Z_DVAL_PP(tmpzval));
	}

	if (!scratch) {
		scratch_len = strlen(path) + 29 + protocol_version_len;
		scratch = emalloc(scratch_len);
		strncpy(scratch, "GET ", scratch_len);
	}

	/* Should we send the entire path in the request line, default to no. */
	if (!request_fulluri &&
		context &&
		php_stream_context_get_option(context, "http", "request_fulluri", &tmpzval) == SUCCESS) {
		zval ztmp = **tmpzval;

		zval_copy_ctor(&ztmp);
		convert_to_boolean(&ztmp);
		request_fulluri = Z_BVAL(ztmp) ? 1 : 0;
		zval_dtor(&ztmp);
	}

	if (request_fulluri) {
		/* Ask for everything */
		strcat(scratch, path);
	} else {
		/* Send the traditional /path/to/file?query_string */

		/* file */
		if (resource->path && *resource->path) {
			strlcat(scratch, resource->path, scratch_len);
		} else {
			strlcat(scratch, "/", scratch_len);
		}

		/* query string */
		if (resource->query) {
			strlcat(scratch, "?", scratch_len);
			strlcat(scratch, resource->query, scratch_len);
		}
	}

	/* protocol version we are speaking */
	if (protocol_version) {
		strlcat(scratch, " HTTP/", scratch_len);
		strlcat(scratch, protocol_version, scratch_len);
		strlcat(scratch, "\r\n", scratch_len);
	} else {
		strlcat(scratch, " HTTP/1.0\r\n", scratch_len);
	}

	/* send it */
	php_stream_write(stream, scratch, strlen(scratch));

	if (context && php_stream_context_get_option(context, "http", "header", &tmpzval) == SUCCESS) {
		tmp = NULL;

		if (Z_TYPE_PP(tmpzval) == IS_ARRAY) {
			HashPosition pos;
			zval **tmpheader = NULL;
			smart_str tmpstr = {0};

			for (zend_hash_internal_pointer_reset_ex(Z_ARRVAL_PP(tmpzval), &pos);
				SUCCESS == zend_hash_get_current_data_ex(Z_ARRVAL_PP(tmpzval), (void *)&tmpheader, &pos);
				zend_hash_move_forward_ex(Z_ARRVAL_PP(tmpzval), &pos)
			) {
				if (Z_TYPE_PP(tmpheader) == IS_STRING) {
					smart_str_appendl(&tmpstr, Z_STRVAL_PP(tmpheader), Z_STRLEN_PP(tmpheader));
					smart_str_appendl(&tmpstr, "\r\n", sizeof("\r\n") - 1);
				}
			}
			smart_str_0(&tmpstr);
			/* Remove newlines and spaces from start and end. there's at least one extra \r\n at the end that needs to go. */
			if (tmpstr.c) {
				tmp = php_trim(tmpstr.c, strlen(tmpstr.c), NULL, 0, NULL, 3 TSRMLS_CC);
				smart_str_free(&tmpstr);
			}
		}
		if (Z_TYPE_PP(tmpzval) == IS_STRING && Z_STRLEN_PP(tmpzval)) {
			/* Remove newlines and spaces from start and end php_trim will estrndup() */
			tmp = php_trim(Z_STRVAL_PP(tmpzval), Z_STRLEN_PP(tmpzval), NULL, 0, NULL, 3 TSRMLS_CC);
		}
		if (tmp && strlen(tmp) > 0) {
			char *s;

			user_headers = estrdup(tmp);

			/* Make lowercase for easy comparison against 'standard' headers */
			php_strtolower(tmp, strlen(tmp));

			if (!header_init) {
				/* strip POST headers on redirect */
				strip_header(user_headers, tmp, "content-length:");
				strip_header(user_headers, tmp, "content-type:");
			}

			if ((s = strstr(tmp, "user-agent:")) &&
			    (s == tmp || *(s-1) == '\r' || *(s-1) == '\n' ||
			                 *(s-1) == '\t' || *(s-1) == ' ')) {
				 have_header |= HTTP_HEADER_USER_AGENT;
			}
			if ((s = strstr(tmp, "host:")) &&
			    (s == tmp || *(s-1) == '\r' || *(s-1) == '\n' ||
			                 *(s-1) == '\t' || *(s-1) == ' ')) {
				 have_header |= HTTP_HEADER_HOST;
			}
			if ((s = strstr(tmp, "from:")) &&
			    (s == tmp || *(s-1) == '\r' || *(s-1) == '\n' ||
			                 *(s-1) == '\t' || *(s-1) == ' ')) {
				 have_header |= HTTP_HEADER_FROM;
				}
			if ((s = strstr(tmp, "authorization:")) &&
			    (s == tmp || *(s-1) == '\r' || *(s-1) == '\n' ||
			                 *(s-1) == '\t' || *(s-1) == ' ')) {
				 have_header |= HTTP_HEADER_AUTH;
			}
			if ((s = strstr(tmp, "content-length:")) &&
			    (s == tmp || *(s-1) == '\r' || *(s-1) == '\n' ||
			                 *(s-1) == '\t' || *(s-1) == ' ')) {
				 have_header |= HTTP_HEADER_CONTENT_LENGTH;
			}
			if ((s = strstr(tmp, "content-type:")) &&
			    (s == tmp || *(s-1) == '\r' || *(s-1) == '\n' ||
			                 *(s-1) == '\t' || *(s-1) == ' ')) {
				 have_header |= HTTP_HEADER_TYPE;
			}
			if ((s = strstr(tmp, "connection:")) &&
			    (s == tmp || *(s-1) == '\r' || *(s-1) == '\n' ||
			                 *(s-1) == '\t' || *(s-1) == ' ')) {
				 have_header |= HTTP_HEADER_CONNECTION;
			}
			/* remove Proxy-Authorization header */
			if (use_proxy && use_ssl && (s = strstr(tmp, "proxy-authorization:")) &&
			    (s == tmp || *(s-1) == '\r' || *(s-1) == '\n' ||
			                 *(s-1) == '\t' || *(s-1) == ' ')) {
				char *p = s + sizeof("proxy-authorization:") - 1;

				while (s > tmp && (*(s-1) == ' ' || *(s-1) == '\t')) s--;
				while (*p != 0 && *p != '\r' && *p != '\n') p++;
				while (*p == '\r' || *p == '\n') p++;
				if (*p == 0) {
					if (s == tmp) {
						efree(user_headers);
						user_headers = NULL;
					} else {
						while (s > tmp && (*(s-1) == '\r' || *(s-1) == '\n')) s--;
						user_headers[s - tmp] = 0;
					}
				} else {
					memmove(user_headers + (s - tmp), user_headers + (p - tmp), strlen(p) + 1);
				}
			}

		}
		if (tmp) {
			efree(tmp);
		}
	}

	/* auth header if it was specified */
	if (((have_header & HTTP_HEADER_AUTH) == 0) && resource->user) {
		/* decode the strings first */
		php_url_decode(resource->user, strlen(resource->user));

		/* scratch is large enough, since it was made large enough for the whole URL */
		strcpy(scratch, resource->user);
		strcat(scratch, ":");

		/* Note: password is optional! */
		if (resource->pass) {
			php_url_decode(resource->pass, strlen(resource->pass));
			strcat(scratch, resource->pass);
		}

		tmp = (char*)php_base64_encode((unsigned char*)scratch, strlen(scratch), NULL);

		if (snprintf(scratch, scratch_len, "Authorization: Basic %s\r\n", tmp) > 0) {
			php_stream_write(stream, scratch, strlen(scratch));
			php_stream_notify_info(context, PHP_STREAM_NOTIFY_AUTH_REQUIRED, NULL, 0);
		}

		efree(tmp);
		tmp = NULL;
	}

	/* if the user has configured who they are, send a From: line */
	if (((have_header & HTTP_HEADER_FROM) == 0) && FG(from_address)) {
		if (snprintf(scratch, scratch_len, "From: %s\r\n", FG(from_address)) > 0)
			php_stream_write(stream, scratch, strlen(scratch));
	}

	/* Send Host: header so name-based virtual hosts work */
	if ((have_header & HTTP_HEADER_HOST) == 0) {
		if ((use_ssl && resource->port != 443 && resource->port != 0) ||
			(!use_ssl && resource->port != 80 && resource->port != 0)) {
			if (snprintf(scratch, scratch_len, "Host: %s:%i\r\n", resource->host, resource->port) > 0)
				php_stream_write(stream, scratch, strlen(scratch));
		} else {
			if (snprintf(scratch, scratch_len, "Host: %s\r\n", resource->host) > 0) {
				php_stream_write(stream, scratch, strlen(scratch));
			}
		}
	}

	/* Send a Connection: close header to avoid hanging when the server
	 * interprets the RFC literally and establishes a keep-alive connection,
	 * unless the user specifically requests something else by specifying a
	 * Connection header in the context options. Send that header even for
	 * HTTP/1.0 to avoid issues when the server respond with a HTTP/1.1
	 * keep-alive response, which is the preferred response type. */
	if ((have_header & HTTP_HEADER_CONNECTION) == 0) {
		php_stream_write_string(stream, "Connection: close\r\n");
	}

	if (context &&
	    php_stream_context_get_option(context, "http", "user_agent", &ua_zval) == SUCCESS &&
		Z_TYPE_PP(ua_zval) == IS_STRING) {
		ua_str = Z_STRVAL_PP(ua_zval);
	} else if (FG(user_agent)) {
		ua_str = FG(user_agent);
	}

	if (((have_header & HTTP_HEADER_USER_AGENT) == 0) && ua_str) {
#define _UA_HEADER "User-Agent: %s\r\n"
		char *ua;
		size_t ua_len;

		ua_len = sizeof(_UA_HEADER) + strlen(ua_str);

		/* ensure the header is only sent if user_agent is not blank */
		if (ua_len > sizeof(_UA_HEADER)) {
			ua = emalloc(ua_len + 1);
			if ((ua_len = slprintf(ua, ua_len, _UA_HEADER, ua_str)) > 0) {
				ua[ua_len] = 0;
				php_stream_write(stream, ua, ua_len);
			} else {
				php_error_docref(NULL TSRMLS_CC, E_WARNING, "Cannot construct User-agent header");
			}

			if (ua) {
				efree(ua);
			}
		}
	}

	if (user_headers) {
		/* A bit weird, but some servers require that Content-Length be sent prior to Content-Type for POST
		 * see bug #44603 for details. Since Content-Type maybe part of user's headers we need to do this check first.
		 */
		if (
				header_init &&
				context &&
				!(have_header & HTTP_HEADER_CONTENT_LENGTH) &&
				php_stream_context_get_option(context, "http", "content", &tmpzval) == SUCCESS &&
				Z_TYPE_PP(tmpzval) == IS_STRING && Z_STRLEN_PP(tmpzval) > 0
		) {
			scratch_len = slprintf(scratch, scratch_len, "Content-Length: %d\r\n", Z_STRLEN_PP(tmpzval));
			php_stream_write(stream, scratch, scratch_len);
			have_header |= HTTP_HEADER_CONTENT_LENGTH;
		}

		php_stream_write(stream, user_headers, strlen(user_headers));
		php_stream_write(stream, "\r\n", sizeof("\r\n")-1);
		efree(user_headers);
	}

	/* Request content, such as for POST requests */
	if (header_init && context &&
		php_stream_context_get_option(context, "http", "content", &tmpzval) == SUCCESS &&
		Z_TYPE_PP(tmpzval) == IS_STRING && Z_STRLEN_PP(tmpzval) > 0) {
		if (!(have_header & HTTP_HEADER_CONTENT_LENGTH)) {
			scratch_len = slprintf(scratch, scratch_len, "Content-Length: %d\r\n", Z_STRLEN_PP(tmpzval));
			php_stream_write(stream, scratch, scratch_len);
		}
		if (!(have_header & HTTP_HEADER_TYPE)) {
			php_stream_write(stream, "Content-Type: application/x-www-form-urlencoded\r\n",
				sizeof("Content-Type: application/x-www-form-urlencoded\r\n") - 1);
			php_error_docref(NULL TSRMLS_CC, E_NOTICE, "Content-type not specified assuming application/x-www-form-urlencoded");
		}
		php_stream_write(stream, "\r\n", sizeof("\r\n")-1);
		php_stream_write(stream, Z_STRVAL_PP(tmpzval), Z_STRLEN_PP(tmpzval));
	} else {
		php_stream_write(stream, "\r\n", sizeof("\r\n")-1);
	}

	location[0] = '\0';

	if (!EG(active_symbol_table)) {
		zend_rebuild_symbol_table(TSRMLS_C);
	}

	if (header_init) {
		zval *ztmp;
		MAKE_STD_ZVAL(ztmp);
		array_init(ztmp);
		ZEND_SET_SYMBOL(EG(active_symbol_table), "http_response_header", ztmp);
	}

	{
		zval **rh;
		if(zend_hash_find(EG(active_symbol_table), "http_response_header", sizeof("http_response_header"), (void **) &rh) != SUCCESS || Z_TYPE_PP(rh) != IS_ARRAY) {
			php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "HTTP request failed, http_response_header overwritten");
			goto out;
		}
		response_header = *rh;
		Z_ADDREF_P(response_header);
	}

	if (!php_stream_eof(stream)) {
		size_t tmp_line_len;
		/* get response header */

		if (php_stream_get_line(stream, tmp_line, sizeof(tmp_line) - 1, &tmp_line_len) != NULL) {
			zval *http_response;

			if (tmp_line_len > 9) {
				response_code = atoi(tmp_line + 9);
			} else {
				response_code = 0;
			}
			if (context && SUCCESS==php_stream_context_get_option(context, "http", "ignore_errors", &tmpzval)) {
				ignore_errors = zend_is_true(*tmpzval);
			}
			/* when we request only the header, don't fail even on error codes */
			if ((options & STREAM_ONLY_GET_HEADERS) || ignore_errors) {
				reqok = 1;
			}

			/* status codes of 1xx are "informational", and will be followed by a real response
			 * e.g "100 Continue". RFC 7231 states that unexpected 1xx status MUST be parsed,
			 * and MAY be ignored. As such, we need to skip ahead to the "real" status*/
			if (response_code >= 100 && response_code < 200) {
				/* consume lines until we find a line starting 'HTTP/1' */
				while (
					!php_stream_eof(stream)
					&& php_stream_get_line(stream, tmp_line, sizeof(tmp_line) - 1, &tmp_line_len) != NULL
					&& ( tmp_line_len < sizeof("HTTP/1") - 1 || strncasecmp(tmp_line, "HTTP/1", sizeof("HTTP/1") - 1) )
				);

				if (tmp_line_len > 9) {
					response_code = atoi(tmp_line + 9);
				} else {
					response_code = 0;
				}
			}
			/* all status codes in the 2xx range are defined by the specification as successful;
			 * all status codes in the 3xx range are for redirection, and so also should never
			 * fail */
			if (response_code >= 200 && response_code < 400) {
				reqok = 1;
			} else {
				switch(response_code) {
					case 403:
						php_stream_notify_error(context, PHP_STREAM_NOTIFY_AUTH_RESULT,
								tmp_line, response_code);
						break;
					default:
						/* safety net in the event tmp_line == NULL */
						if (!tmp_line_len) {
							tmp_line[0] = '\0';
						}
						php_stream_notify_error(context, PHP_STREAM_NOTIFY_FAILURE,
 								tmp_line, response_code);
 				}
 			}
			if (tmp_line_len >= 1 && tmp_line[tmp_line_len - 1] == '\n') {
 				--tmp_line_len;
				if (tmp_line_len >= 1 &&tmp_line[tmp_line_len - 1] == '\r') {
 					--tmp_line_len;
 				}
 			}
			MAKE_STD_ZVAL(http_response);
			ZVAL_STRINGL(http_response, tmp_line, tmp_line_len, 1);
			zend_hash_next_index_insert(Z_ARRVAL_P(response_header), &http_response, sizeof(zval *), NULL);
		}
	} else {
		php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "HTTP request failed, unexpected end of socket!");
		goto out;
	}

	/* read past HTTP headers */

	http_header_line = emalloc(HTTP_HEADER_BLOCK_SIZE);

	while (!body && !php_stream_eof(stream)) {
		size_t http_header_line_length;
		if (php_stream_get_line(stream, http_header_line, HTTP_HEADER_BLOCK_SIZE, &http_header_line_length) && *http_header_line != '\n' && *http_header_line != '\r') {
			char *e = http_header_line + http_header_line_length - 1;
			if (*e != '\n') {
				do { /* partial header */
					if (php_stream_get_line(stream, http_header_line, HTTP_HEADER_BLOCK_SIZE, &http_header_line_length) == NULL) {
						php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "Failed to read HTTP headers");
						goto out;
					}
					e = http_header_line + http_header_line_length - 1;
				} while (*e != '\n');
				continue;
			}
			while (*e == '\n' || *e == '\r') {
				e--;
			}
			http_header_line_length = e - http_header_line + 1;
			http_header_line[http_header_line_length] = '\0';

			if (!strncasecmp(http_header_line, "Location: ", 10)) {
				if (context && php_stream_context_get_option(context, "http", "follow_location", &tmpzval) == SUCCESS) {
					SEPARATE_ZVAL(tmpzval);
					convert_to_long_ex(tmpzval);
					follow_location = Z_LVAL_PP(tmpzval);
				} else if (!(response_code >= 300 && response_code < 304 || 307 == response_code || 308 == response_code)) {
					/* we shouldn't redirect automatically
					if follow_location isn't set and response_code not in (300, 301, 302, 303 and 307)
					see http://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html#sec10.3.1
					RFC 7238 defines 308: http://tools.ietf.org/html/rfc7238 */
					follow_location = 0;
				}
				strlcpy(location, http_header_line + 10, sizeof(location));
			} else if (!strncasecmp(http_header_line, "Content-Type: ", 14)) {
				php_stream_notify_info(context, PHP_STREAM_NOTIFY_MIME_TYPE_IS, http_header_line + 14, 0);
			} else if (!strncasecmp(http_header_line, "Content-Length: ", 16)) {
				file_size = atoi(http_header_line + 16);
				php_stream_notify_file_size(context, file_size, http_header_line, 0);
			} else if (!strncasecmp(http_header_line, "Transfer-Encoding: chunked", sizeof("Transfer-Encoding: chunked"))) {

				/* create filter to decode response body */
				if (!(options & STREAM_ONLY_GET_HEADERS)) {
					long decode = 1;

					if (context && php_stream_context_get_option(context, "http", "auto_decode", &tmpzval) == SUCCESS) {
						SEPARATE_ZVAL(tmpzval);
						convert_to_boolean(*tmpzval);
						decode = Z_LVAL_PP(tmpzval);
					}
					if (decode) {
						transfer_encoding = php_stream_filter_create("dechunk", NULL, php_stream_is_persistent(stream) TSRMLS_CC);
						if (transfer_encoding) {
							/* don't store transfer-encodeing header */
							continue;
						}
					}
				}
			}

			if (http_header_line[0] == '\0') {
				body = 1;
			} else {
				zval *http_header;

				MAKE_STD_ZVAL(http_header);

				ZVAL_STRINGL(http_header, http_header_line, http_header_line_length, 1);

				zend_hash_next_index_insert(Z_ARRVAL_P(response_header), &http_header, sizeof(zval *), NULL);
			}
		} else {
			break;
		}
	}

	if (!reqok || (location[0] != '\0' && follow_location)) {
		if (!follow_location || (((options & STREAM_ONLY_GET_HEADERS) || ignore_errors) && redirect_max <= 1)) {
			goto out;
		}

		if (location[0] != '\0')
			php_stream_notify_info(context, PHP_STREAM_NOTIFY_REDIRECTED, location, 0);

		php_stream_close(stream);
		stream = NULL;

		if (location[0] != '\0') {

			char new_path[HTTP_HEADER_BLOCK_SIZE];
			char loc_path[HTTP_HEADER_BLOCK_SIZE];

			*new_path='\0';
			if (strlen(location)<8 || (strncasecmp(location, "http://", sizeof("http://")-1) &&
							strncasecmp(location, "https://", sizeof("https://")-1) &&
							strncasecmp(location, "ftp://", sizeof("ftp://")-1) &&
							strncasecmp(location, "ftps://", sizeof("ftps://")-1)))
			{
				if (*location != '/') {
					if (*(location+1) != '\0' && resource->path) {
						char *s = strrchr(resource->path, '/');
						if (!s) {
							s = resource->path;
							if (!s[0]) {
								efree(s);
								s = resource->path = estrdup("/");
							} else {
								*s = '/';
							}
						}
						s[1] = '\0';
						if (resource->path && *(resource->path) == '/' && *(resource->path + 1) == '\0') {
							snprintf(loc_path, sizeof(loc_path) - 1, "%s%s", resource->path, location);
						} else {
							snprintf(loc_path, sizeof(loc_path) - 1, "%s/%s", resource->path, location);
						}
					} else {
						snprintf(loc_path, sizeof(loc_path) - 1, "/%s", location);
					}
				} else {
					strlcpy(loc_path, location, sizeof(loc_path));
				}
				if ((use_ssl && resource->port != 443) || (!use_ssl && resource->port != 80)) {
					snprintf(new_path, sizeof(new_path) - 1, "%s://%s:%d%s", resource->scheme, resource->host, resource->port, loc_path);
				} else {
					snprintf(new_path, sizeof(new_path) - 1, "%s://%s%s", resource->scheme, resource->host, loc_path);
				}
			} else {
				strlcpy(new_path, location, sizeof(new_path));
			}

			php_url_free(resource);
			/* check for invalid redirection URLs */
			if ((resource = php_url_parse(new_path)) == NULL) {
				php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "Invalid redirect URL! %s", new_path);
				goto out;
			}

#define CHECK_FOR_CNTRL_CHARS(val) { \
	if (val) { \
		unsigned char *s, *e; \
		int l; \
		l = php_url_decode(val, strlen(val)); \
		s = (unsigned char*)val; e = s + l; \
		while (s < e) { \
			if (iscntrl(*s)) { \
				php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "Invalid redirect URL! %s", new_path); \
				goto out; \
			} \
			s++; \
		} \
	} \
}
			/* check for control characters in login, password & path */
			if (strncasecmp(new_path, "http://", sizeof("http://") - 1) || strncasecmp(new_path, "https://", sizeof("https://") - 1)) {
				CHECK_FOR_CNTRL_CHARS(resource->user)
				CHECK_FOR_CNTRL_CHARS(resource->pass)
				CHECK_FOR_CNTRL_CHARS(resource->path)
			}
			stream = php_stream_url_wrap_http_ex(wrapper, new_path, mode, options, opened_path, context, --redirect_max, HTTP_WRAPPER_REDIRECTED STREAMS_CC TSRMLS_CC);
		} else {
			php_stream_wrapper_log_error(wrapper, options TSRMLS_CC, "HTTP request failed! %s", tmp_line);
		}
	}
out:
	if (protocol_version) {
		efree(protocol_version);
	}

	if (http_header_line) {
		efree(http_header_line);
	}

	if (scratch) {
		efree(scratch);
	}

	if (resource) {
		php_url_free(resource);
	}

	if (stream) {
		if (header_init) {
			stream->wrapperdata = response_header;
		} else {
			if(response_header) {
				Z_DELREF_P(response_header);
			}
		}
		php_stream_notify_progress_init(context, 0, file_size);

		/* Restore original chunk size now that we're done with headers */
		if (options & STREAM_WILL_CAST)
			php_stream_set_chunk_size(stream, chunk_size);

		/* restore the users auto-detect-line-endings setting */
		stream->flags |= eol_detect;

		/* as far as streams are concerned, we are now at the start of
		 * the stream */
		stream->position = 0;

		/* restore mode */
		strlcpy(stream->mode, mode, sizeof(stream->mode));

		if (transfer_encoding) {
			php_stream_filter_append(&stream->readfilters, transfer_encoding);
		}
	} else {
		if(response_header) {
			Z_DELREF_P(response_header);
		}
		if (transfer_encoding) {
			php_stream_filter_free(transfer_encoding TSRMLS_CC);
		}
	}

	return stream;
}
/* }}} */
```

---

## Pair 2 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `devign_train_13191`
- **Project:** `qemu`
- **Commit ID:** `de1c90cf8b73992b4197561850d2da1075fb82eb`
- **CWE:** `None`
- **Tokens:** 34

```c
static void qemu_event_increment(void)

{

    SetEvent(qemu_event_handle);

}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_211523`
- **Project:** `LoRaMac-node`
- **Commit ID:** `e3063a91daa7ad8a687223efa63079f0c24568e4`
- **Tokens:** 47

```c
static void EventSecureElementNvmCtxChanged( void )
{
    CallNvmCtxCallback( LORAMAC_NVMCTXMODULE_SECURE_ELEMENT );
}
```

---

## Pair 3 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `dv_train_171977`
- **Project:** `FFmpeg`
- **Commit ID:** `fe448cd28d674c3eff3072552eae366d0b659ce9`
- **CWE:** `CWE-119`
- **Tokens:** 4172

```c
static int jpeg2000_decode_tile(Jpeg2000DecoderContext *s, Jpeg2000Tile *tile,
                                AVFrame *picture)
{
    int compno, reslevelno, bandno;
    int x, y;

    uint8_t *line;
    Jpeg2000T1Context t1;

    /* Loop on tile components */
    for (compno = 0; compno < s->ncomponents; compno++) {
        Jpeg2000Component *comp     = tile->comp + compno;
        Jpeg2000CodingStyle *codsty = tile->codsty + compno;

        /* Loop on resolution levels */
        for (reslevelno = 0; reslevelno < codsty->nreslevels2decode; reslevelno++) {
            Jpeg2000ResLevel *rlevel = comp->reslevel + reslevelno;
            /* Loop on bands */
            for (bandno = 0; bandno < rlevel->nbands; bandno++) {
                int nb_precincts, precno;
                Jpeg2000Band *band = rlevel->band + bandno;
                int cblkno = 0, bandpos;

                bandpos = bandno + (reslevelno > 0);

                if (band->coord[0][0] == band->coord[0][1] ||
                    band->coord[1][0] == band->coord[1][1])
                    continue;

                nb_precincts = rlevel->num_precincts_x * rlevel->num_precincts_y;
                /* Loop on precincts */
                for (precno = 0; precno < nb_precincts; precno++) {
                    Jpeg2000Prec *prec = band->prec + precno;

                    /* Loop on codeblocks */
                    for (cblkno = 0; cblkno < prec->nb_codeblocks_width * prec->nb_codeblocks_height; cblkno++) {
                        int x, y;
                        Jpeg2000Cblk *cblk = prec->cblk + cblkno;
                        decode_cblk(s, codsty, &t1, cblk,
                                    cblk->coord[0][1] - cblk->coord[0][0],
                                    cblk->coord[1][1] - cblk->coord[1][0],
                                    bandpos);

                        x = cblk->coord[0][0];
                        y = cblk->coord[1][0];

                        if (codsty->transform == FF_DWT97)
                            dequantization_float(x, y, cblk, comp, &t1, band);
                        else
                            dequantization_int(x, y, cblk, comp, &t1, band);
                   } /* end cblk */
                } /*end prec */
            } /* end band */
        } /* end reslevel */

        /* inverse DWT */
        ff_dwt_decode(&comp->dwt, codsty->transform == FF_DWT97 ? (void*)comp->f_data : (void*)comp->i_data);
    } /*end comp */

    /* inverse MCT transformation */
    if (tile->codsty[0].mct)
        mct_decode(s, tile);

    if (s->cdef[0] < 0) {
        for (x = 0; x < s->ncomponents; x++)
            s->cdef[x] = x + 1;
        if ((s->ncomponents & 1) == 0)
            s->cdef[s->ncomponents-1] = 0;
    }

    if (s->precision <= 8) {
        for (compno = 0; compno < s->ncomponents; compno++) {
            Jpeg2000Component *comp = tile->comp + compno;
            Jpeg2000CodingStyle *codsty = tile->codsty + compno;
            float *datap = comp->f_data;
            int32_t *i_datap = comp->i_data;
            int cbps = s->cbps[compno];
            int w = tile->comp[compno].coord[0][1] - s->image_offset_x;
            int planar = !!picture->data[2];
            int pixelsize = planar ? 1 : s->ncomponents;
            int plane = 0;

            if (planar)
                plane = s->cdef[compno] ? s->cdef[compno]-1 : (s->ncomponents-1);


            y    = tile->comp[compno].coord[1][0] - s->image_offset_y;
            line = picture->data[plane] + y * picture->linesize[plane];
            for (; y < tile->comp[compno].coord[1][1] - s->image_offset_y; y += s->cdy[compno]) {
                uint8_t *dst;

                x   = tile->comp[compno].coord[0][0] - s->image_offset_x;
                dst = line + x * pixelsize + compno*!planar;

                if (codsty->transform == FF_DWT97) {
                    for (; x < w; x += s->cdx[compno]) {
                        int val = lrintf(*datap) + (1 << (cbps - 1));
                        /* DC level shift and clip see ISO 15444-1:2002 G.1.2 */
                        val = av_clip(val, 0, (1 << cbps) - 1);
                        *dst = val << (8 - cbps);
                        datap++;
                        dst += pixelsize;
                    }
                } else {
                    for (; x < w; x += s->cdx[compno]) {
                        int val = *i_datap + (1 << (cbps - 1));
                        /* DC level shift and clip see ISO 15444-1:2002 G.1.2 */
                        val = av_clip(val, 0, (1 << cbps) - 1);
                        *dst = val << (8 - cbps);
                        i_datap++;
                        dst += pixelsize;
                    }
                }
                line += picture->linesize[plane];
            }
        }
    } else {
        for (compno = 0; compno < s->ncomponents; compno++) {
            Jpeg2000Component *comp = tile->comp + compno;
            Jpeg2000CodingStyle *codsty = tile->codsty + compno;
            float *datap = comp->f_data;
            int32_t *i_datap = comp->i_data;
            uint16_t *linel;
            int cbps = s->cbps[compno];
            int w = tile->comp[compno].coord[0][1] - s->image_offset_x;
            int planar = !!picture->data[2];
            int pixelsize = planar ? 1 : s->ncomponents;
            int plane = 0;

            if (planar)
                plane = s->cdef[compno] ? s->cdef[compno]-1 : (s->ncomponents-1);

            y     = tile->comp[compno].coord[1][0] - s->image_offset_y;
            linel = (uint16_t *)picture->data[plane] + y * (picture->linesize[plane] >> 1);
            for (; y < tile->comp[compno].coord[1][1] - s->image_offset_y; y += s->cdy[compno]) {
                uint16_t *dst;

                x   = tile->comp[compno].coord[0][0] - s->image_offset_x;
                dst = linel + (x * pixelsize + compno*!planar);
                if (codsty->transform == FF_DWT97) {
                    for (; x < w; x += s-> cdx[compno]) {
                        int  val = lrintf(*datap) + (1 << (cbps - 1));
                        /* DC level shift and clip see ISO 15444-1:2002 G.1.2 */
                        val = av_clip(val, 0, (1 << cbps) - 1);
                        /* align 12 bit values in little-endian mode */
                        *dst = val << (16 - cbps);
                        datap++;
                        dst += pixelsize;
                    }
                } else {
                    for (; x < w; x += s-> cdx[compno]) {
                        int val = *i_datap + (1 << (cbps - 1));
                        /* DC level shift and clip see ISO 15444-1:2002 G.1.2 */
                        val = av_clip(val, 0, (1 << cbps) - 1);
                        /* align 12 bit values in little-endian mode */
                        *dst = val << (16 - cbps);
                        i_datap++;
                        dst += pixelsize;
                    }
                }
                linel += picture->linesize[plane] >> 1;
            }
        }
    }

    return 0;
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `devign_test_15420`
- **Project:** `FFmpeg`
- **Commit ID:** `fe448cd28d674c3eff3072552eae366d0b659ce9`
- **Tokens:** 4195

```c
static int jpeg2000_decode_tile(Jpeg2000DecoderContext *s, Jpeg2000Tile *tile,

                                AVFrame *picture)

{

    int compno, reslevelno, bandno;

    int x, y;



    uint8_t *line;

    Jpeg2000T1Context t1;



    /* Loop on tile components */

    for (compno = 0; compno < s->ncomponents; compno++) {

        Jpeg2000Component *comp     = tile->comp + compno;

        Jpeg2000CodingStyle *codsty = tile->codsty + compno;



        /* Loop on resolution levels */

        for (reslevelno = 0; reslevelno < codsty->nreslevels2decode; reslevelno++) {

            Jpeg2000ResLevel *rlevel = comp->reslevel + reslevelno;

            /* Loop on bands */

            for (bandno = 0; bandno < rlevel->nbands; bandno++) {

                int nb_precincts, precno;

                Jpeg2000Band *band = rlevel->band + bandno;

                int cblkno = 0, bandpos;



                bandpos = bandno + (reslevelno > 0);



                if (band->coord[0][0] == band->coord[0][1] ||

                    band->coord[1][0] == band->coord[1][1])

                    continue;



                nb_precincts = rlevel->num_precincts_x * rlevel->num_precincts_y;

                /* Loop on precincts */

                for (precno = 0; precno < nb_precincts; precno++) {

                    Jpeg2000Prec *prec = band->prec + precno;



                    /* Loop on codeblocks */

                    for (cblkno = 0; cblkno < prec->nb_codeblocks_width * prec->nb_codeblocks_height; cblkno++) {

                        int x, y;

                        Jpeg2000Cblk *cblk = prec->cblk + cblkno;

                        decode_cblk(s, codsty, &t1, cblk,

                                    cblk->coord[0][1] - cblk->coord[0][0],

                                    cblk->coord[1][1] - cblk->coord[1][0],

                                    bandpos);



                        x = cblk->coord[0][0];

                        y = cblk->coord[1][0];



                        if (codsty->transform == FF_DWT97)

                            dequantization_float(x, y, cblk, comp, &t1, band);

                        else

                            dequantization_int(x, y, cblk, comp, &t1, band);

                   } /* end cblk */

                } /*end prec */

            } /* end band */

        } /* end reslevel */



        /* inverse DWT */

        ff_dwt_decode(&comp->dwt, codsty->transform == FF_DWT97 ? (void*)comp->f_data : (void*)comp->i_data);

    } /*end comp */



    /* inverse MCT transformation */

    if (tile->codsty[0].mct)

        mct_decode(s, tile);



    if (s->cdef[0] < 0) {

        for (x = 0; x < s->ncomponents; x++)

            s->cdef[x] = x + 1;

        if ((s->ncomponents & 1) == 0)

            s->cdef[s->ncomponents-1] = 0;

    }



    if (s->precision <= 8) {

        for (compno = 0; compno < s->ncomponents; compno++) {

            Jpeg2000Component *comp = tile->comp + compno;

            Jpeg2000CodingStyle *codsty = tile->codsty + compno;

            float *datap = comp->f_data;

            int32_t *i_datap = comp->i_data;

            int cbps = s->cbps[compno];

            int w = tile->comp[compno].coord[0][1] - s->image_offset_x;

            int planar = !!picture->data[2];

            int pixelsize = planar ? 1 : s->ncomponents;

            int plane = 0;



            if (planar)

                plane = s->cdef[compno] ? s->cdef[compno]-1 : (s->ncomponents-1);





            y    = tile->comp[compno].coord[1][0] - s->image_offset_y;

            line = picture->data[plane] + y * picture->linesize[plane];

            for (; y < tile->comp[compno].coord[1][1] - s->image_offset_y; y += s->cdy[compno]) {

                uint8_t *dst;



                x   = tile->comp[compno].coord[0][0] - s->image_offset_x;

                dst = line + x * pixelsize + compno*!planar;



                if (codsty->transform == FF_DWT97) {

                    for (; x < w; x += s->cdx[compno]) {

                        int val = lrintf(*datap) + (1 << (cbps - 1));

                        /* DC level shift and clip see ISO 15444-1:2002 G.1.2 */

                        val = av_clip(val, 0, (1 << cbps) - 1);

                        *dst = val << (8 - cbps);

                        datap++;

                        dst += pixelsize;

                    }

                } else {

                    for (; x < w; x += s->cdx[compno]) {

                        int val = *i_datap + (1 << (cbps - 1));

                        /* DC level shift and clip see ISO 15444-1:2002 G.1.2 */

                        val = av_clip(val, 0, (1 << cbps) - 1);

                        *dst = val << (8 - cbps);

                        i_datap++;

                        dst += pixelsize;

                    }

                }

                line += picture->linesize[plane];

            }

        }

    } else {

        for (compno = 0; compno < s->ncomponents; compno++) {

            Jpeg2000Component *comp = tile->comp + compno;

            Jpeg2000CodingStyle *codsty = tile->codsty + compno;

            float *datap = comp->f_data;

            int32_t *i_datap = comp->i_data;

            uint16_t *linel;

            int cbps = s->cbps[compno];

            int w = tile->comp[compno].coord[0][1] - s->image_offset_x;

            int planar = !!picture->data[2];

            int pixelsize = planar ? 1 : s->ncomponents;

            int plane = 0;



            if (planar)

                plane = s->cdef[compno] ? s->cdef[compno]-1 : (s->ncomponents-1);



            y     = tile->comp[compno].coord[1][0] - s->image_offset_y;

            linel = (uint16_t *)picture->data[plane] + y * (picture->linesize[plane] >> 1);

            for (; y < tile->comp[compno].coord[1][1] - s->image_offset_y; y += s->cdy[compno]) {

                uint16_t *dst;



                x   = tile->comp[compno].coord[0][0] - s->image_offset_x;

                dst = linel + (x * pixelsize + compno*!planar);

                if (codsty->transform == FF_DWT97) {

                    for (; x < w; x += s-> cdx[compno]) {

                        int  val = lrintf(*datap) + (1 << (cbps - 1));

                        /* DC level shift and clip see ISO 15444-1:2002 G.1.2 */

                        val = av_clip(val, 0, (1 << cbps) - 1);

                        /* align 12 bit values in little-endian mode */

                        *dst = val << (16 - cbps);

                        datap++;

                        dst += pixelsize;

                    }

                } else {

                    for (; x < w; x += s-> cdx[compno]) {

                        int val = *i_datap + (1 << (cbps - 1));

                        /* DC level shift and clip see ISO 15444-1:2002 G.1.2 */

                        val = av_clip(val, 0, (1 << cbps) - 1);

                        /* align 12 bit values in little-endian mode */

                        *dst = val << (16 - cbps);

                        i_datap++;

                        dst += pixelsize;

                    }

                }

                linel += picture->linesize[plane] >> 1;

            }

        }

    }



    return 0;

}
```

---

## Pair 4 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `bv_train_149288`
- **Project:** `ImageMagick`
- **Commit ID:** `d9b2209a69ee90d8df81fb124eb66f593eb9f599`
- **CWE:** `CWE-787`
- **Tokens:** 11454

```c
static Image *ReadWPGImage(const ImageInfo *image_info,
  ExceptionInfo *exception)
{
  typedef struct
  {
    size_t FileId;
    MagickOffsetType DataOffset;
    unsigned int ProductType;
    unsigned int FileType;
    unsigned char MajorVersion;
    unsigned char MinorVersion;
    unsigned int EncryptKey;
    unsigned int Reserved;
  } WPGHeader;

  typedef struct
  {
    unsigned char RecType;
    size_t RecordLength;
  } WPGRecord;

  typedef struct
  {
    unsigned char Class;
    unsigned char RecType;
    size_t Extension;
    size_t RecordLength;
  } WPG2Record;

  typedef struct
  {
    unsigned  HorizontalUnits;
    unsigned  VerticalUnits;
    unsigned char PosSizePrecision;
  } WPG2Start;

  typedef struct
  {
    unsigned int Width;
    unsigned int Height;
    unsigned int Depth;
    unsigned int HorzRes;
    unsigned int VertRes;
  } WPGBitmapType1;

  typedef struct
  {
    unsigned int Width;
    unsigned int Height;
    unsigned char Depth;
    unsigned char Compression;
  } WPG2BitmapType1;

  typedef struct
  {
    unsigned int RotAngle;
    unsigned int LowLeftX;
    unsigned int LowLeftY;
    unsigned int UpRightX;
    unsigned int UpRightY;
    unsigned int Width;
    unsigned int Height;
    unsigned int Depth;
    unsigned int HorzRes;
    unsigned int VertRes;
  } WPGBitmapType2;

  typedef struct
  {
    unsigned int StartIndex;
    unsigned int NumOfEntries;
  } WPGColorMapRec;

  /*
  typedef struct {
    size_t PS_unknown1;
    unsigned int PS_unknown2;
    unsigned int PS_unknown3;
  } WPGPSl1Record;
  */

  Image
    *image;

  unsigned int
    status;

  WPGHeader
    Header;

  WPGRecord
    Rec;

  WPG2Record
    Rec2;

  WPG2Start StartWPG;

  WPGBitmapType1
    BitmapHeader1;

  WPG2BitmapType1
    Bitmap2Header1;

  WPGBitmapType2
    BitmapHeader2;

  WPGColorMapRec
    WPG_Palette;

  int
    i,
    bpp,
    WPG2Flags;

  ssize_t
    ldblk;

  size_t
    one;

  unsigned char
    *BImgBuff;

  tCTM CTM;         /*current transform matrix*/

  /*
    Open image file.
  */
  assert(image_info != (const ImageInfo *) NULL);
  assert(image_info->signature == MagickSignature);
  assert(exception != (ExceptionInfo *) NULL);
  assert(exception->signature == MagickSignature);
  one=1;
  image=AcquireImage(image_info);
  image->depth=8;
  status=OpenBlob(image_info,image,ReadBinaryBlobMode,exception);
  if (status == MagickFalse)
    {
      image=DestroyImageList(image);
      return((Image *) NULL);
    }
  /*
    Read WPG image.
  */
  Header.FileId=ReadBlobLSBLong(image);
  Header.DataOffset=(MagickOffsetType) ReadBlobLSBLong(image);
  Header.ProductType=ReadBlobLSBShort(image);
  Header.FileType=ReadBlobLSBShort(image);
  Header.MajorVersion=ReadBlobByte(image);
  Header.MinorVersion=ReadBlobByte(image);
  Header.EncryptKey=ReadBlobLSBShort(image);
  Header.Reserved=ReadBlobLSBShort(image);

  if (Header.FileId!=0x435057FF || (Header.ProductType>>8)!=0x16)
    ThrowReaderException(CorruptImageError,"ImproperImageHeader");
  if (Header.EncryptKey!=0)
    ThrowReaderException(CoderError,"EncryptedWPGImageFileNotSupported");

  image->columns = 1;
  image->rows = 1;
  image->colors = 0;
  bpp=0;
  BitmapHeader2.RotAngle=0;

  switch(Header.FileType)
    {
    case 1:     /* WPG level 1 */
      while(!EOFBlob(image)) /* object parser loop */
        {
          (void) SeekBlob(image,Header.DataOffset,SEEK_SET);
          if(EOFBlob(image))
            break;

          Rec.RecType=(i=ReadBlobByte(image));
          if(i==EOF)
            break;
          Rd_WP_DWORD(image,&Rec.RecordLength);
          if(EOFBlob(image))
            break;

          Header.DataOffset=TellBlob(image)+Rec.RecordLength;

          switch(Rec.RecType)
            {
            case 0x0B: /* bitmap type 1 */
              BitmapHeader1.Width=ReadBlobLSBShort(image);
              BitmapHeader1.Height=ReadBlobLSBShort(image);
              if ((BitmapHeader1.Width == 0) || (BitmapHeader1.Height == 0))
                ThrowReaderException(CorruptImageError,"ImproperImageHeader");
              BitmapHeader1.Depth=ReadBlobLSBShort(image);
              BitmapHeader1.HorzRes=ReadBlobLSBShort(image);
              BitmapHeader1.VertRes=ReadBlobLSBShort(image);

              if(BitmapHeader1.HorzRes && BitmapHeader1.VertRes)
                {
                  image->units=PixelsPerCentimeterResolution;
                  image->x_resolution=BitmapHeader1.HorzRes/470.0;
                  image->y_resolution=BitmapHeader1.VertRes/470.0;
                }
              image->columns=BitmapHeader1.Width;
              image->rows=BitmapHeader1.Height;
              bpp=BitmapHeader1.Depth;

              goto UnpackRaster;

            case 0x0E:  /*Color palette */
              WPG_Palette.StartIndex=ReadBlobLSBShort(image);
              WPG_Palette.NumOfEntries=ReadBlobLSBShort(image);

              image->colors=WPG_Palette.NumOfEntries;
              if (!AcquireImageColormap(image,image->colors))
                goto NoMemory;
              for (i=WPG_Palette.StartIndex;
                   i < (int)WPG_Palette.NumOfEntries; i++)
                {
                  image->colormap[i].red=ScaleCharToQuantum((unsigned char)
                    ReadBlobByte(image));
                  image->colormap[i].green=ScaleCharToQuantum((unsigned char)
                    ReadBlobByte(image));
                  image->colormap[i].blue=ScaleCharToQuantum((unsigned char)
                    ReadBlobByte(image));
                }
              break;

            case 0x11:  /* Start PS l1 */
              if(Rec.RecordLength > 8)
                image=ExtractPostscript(image,image_info,
                  TellBlob(image)+8,   /* skip PS header in the wpg */
                  (ssize_t) Rec.RecordLength-8,exception);
              break;

            case 0x14:  /* bitmap type 2 */
              BitmapHeader2.RotAngle=ReadBlobLSBShort(image);
              BitmapHeader2.LowLeftX=ReadBlobLSBShort(image);
              BitmapHeader2.LowLeftY=ReadBlobLSBShort(image);
              BitmapHeader2.UpRightX=ReadBlobLSBShort(image);
              BitmapHeader2.UpRightY=ReadBlobLSBShort(image);
              BitmapHeader2.Width=ReadBlobLSBShort(image);
              BitmapHeader2.Height=ReadBlobLSBShort(image);
              if ((BitmapHeader2.Width == 0) || (BitmapHeader2.Height == 0))
                ThrowReaderException(CorruptImageError,"ImproperImageHeader");
              BitmapHeader2.Depth=ReadBlobLSBShort(image);
              BitmapHeader2.HorzRes=ReadBlobLSBShort(image);
              BitmapHeader2.VertRes=ReadBlobLSBShort(image);

              image->units=PixelsPerCentimeterResolution;
              image->page.width=(unsigned int)
                ((BitmapHeader2.LowLeftX-BitmapHeader2.UpRightX)/470.0);
              image->page.height=(unsigned int)
                ((BitmapHeader2.LowLeftX-BitmapHeader2.UpRightY)/470.0);
              image->page.x=(int) (BitmapHeader2.LowLeftX/470.0);
              image->page.y=(int) (BitmapHeader2.LowLeftX/470.0);
              if(BitmapHeader2.HorzRes && BitmapHeader2.VertRes)
                {
                  image->x_resolution=BitmapHeader2.HorzRes/470.0;
                  image->y_resolution=BitmapHeader2.VertRes/470.0;
                }
              image->columns=BitmapHeader2.Width;
              image->rows=BitmapHeader2.Height;
              bpp=BitmapHeader2.Depth;

            UnpackRaster:
              if ((image->colors == 0) && (bpp != 24))
                {
                  image->colors=one << bpp;
                  if (!AcquireImageColormap(image,image->colors))
                    {
                    NoMemory:
                      ThrowReaderException(ResourceLimitError,
                        "MemoryAllocationFailed");
                    }
                  /* printf("Load default colormap \n"); */
                  for (i=0; (i < (int) image->colors) && (i < 256); i++)
                    {
                      image->colormap[i].red=ScaleCharToQuantum(WPG1_Palette[i].Red);
                      image->colormap[i].green=ScaleCharToQuantum(WPG1_Palette[i].Green);
                      image->colormap[i].blue=ScaleCharToQuantum(WPG1_Palette[i].Blue);
                    }
                }
              else
                {
                  if (bpp < 24)
                    if ( (image->colors < (one << bpp)) && (bpp != 24) )
                      image->colormap=(PixelPacket *) ResizeQuantumMemory(
                        image->colormap,(size_t) (one << bpp),
                        sizeof(*image->colormap));
                }

              if (bpp == 1)
                {
                  if(image->colormap[0].red==0 &&
                     image->colormap[0].green==0 &&
                     image->colormap[0].blue==0 &&
                     image->colormap[1].red==0 &&
                     image->colormap[1].green==0 &&
                     image->colormap[1].blue==0)
                    {  /* fix crippled monochrome palette */
                      image->colormap[1].red =
                        image->colormap[1].green =
                        image->colormap[1].blue = QuantumRange;
                    }
                }

              if(UnpackWPGRaster(image,bpp) < 0)
                /* The raster cannot be unpacked */
                {
                DecompressionFailed:
                  ThrowReaderException(CoderError,"UnableToDecompressImage");
                    }

              if(Rec.RecType==0x14 && BitmapHeader2.RotAngle!=0 && !image_info->ping)
                {
                  /* flop command */
                  if(BitmapHeader2.RotAngle & 0x8000)
                    {
                      Image
                        *flop_image;

                      flop_image = FlopImage(image, exception);
                      if (flop_image != (Image *) NULL) {
                        DuplicateBlob(flop_image,image);
                        (void) RemoveLastImageFromList(&image);
                        AppendImageToList(&image,flop_image);
                      }
                    }
                  /* flip command */
                  if(BitmapHeader2.RotAngle & 0x2000)
                    {
                      Image
                        *flip_image;

                      flip_image = FlipImage(image, exception);
                      if (flip_image != (Image *) NULL) {
                        DuplicateBlob(flip_image,image);
                        (void) RemoveLastImageFromList(&image);
                         AppendImageToList(&image,flip_image);
                       }
                     }
      /* rotate command */
                   if(BitmapHeader2.RotAngle & 0x0FFF)
                     {
                       Image
                        *rotate_image;

                      rotate_image=RotateImage(image,(BitmapHeader2.RotAngle &
                        0x0FFF), exception);
                      if (rotate_image != (Image *) NULL) {
                        DuplicateBlob(rotate_image,image);
                        (void) RemoveLastImageFromList(&image);
                        AppendImageToList(&image,rotate_image);
                      }
                    }
                }

              /* Allocate next image structure. */
              AcquireNextImage(image_info,image);
              image->depth=8;
               if (image->next == (Image *) NULL)
                 goto Finish;
               image=SyncNextImageInList(image);
              image->columns=image->rows=0;
               image->colors=0;
               break;

            case 0x1B:  /* Postscript l2 */
              if(Rec.RecordLength>0x3C)
                image=ExtractPostscript(image,image_info,
                  TellBlob(image)+0x3C,   /* skip PS l2 header in the wpg */
                  (ssize_t) Rec.RecordLength-0x3C,exception);
              break;
            }
        }
      break;

    case 2:  /* WPG level 2 */
      (void) memset(CTM,0,sizeof(CTM));
      StartWPG.PosSizePrecision = 0;
      while(!EOFBlob(image)) /* object parser loop */
        {
          (void) SeekBlob(image,Header.DataOffset,SEEK_SET);
          if(EOFBlob(image))
            break;

          Rec2.Class=(i=ReadBlobByte(image));
          if(i==EOF)
            break;
          Rec2.RecType=(i=ReadBlobByte(image));
          if(i==EOF)
            break;
          Rd_WP_DWORD(image,&Rec2.Extension);
          Rd_WP_DWORD(image,&Rec2.RecordLength);
          if(EOFBlob(image))
            break;

          Header.DataOffset=TellBlob(image)+Rec2.RecordLength;

          switch(Rec2.RecType)
            {
      case 1:
              StartWPG.HorizontalUnits=ReadBlobLSBShort(image);
              StartWPG.VerticalUnits=ReadBlobLSBShort(image);
              StartWPG.PosSizePrecision=ReadBlobByte(image);
              break;
            case 0x0C:    /* Color palette */
              WPG_Palette.StartIndex=ReadBlobLSBShort(image);
              WPG_Palette.NumOfEntries=ReadBlobLSBShort(image);

              image->colors=WPG_Palette.NumOfEntries;
              if (AcquireImageColormap(image,image->colors) == MagickFalse)
                ThrowReaderException(ResourceLimitError,
                  "MemoryAllocationFailed");
              for (i=WPG_Palette.StartIndex;
                   i < (int)WPG_Palette.NumOfEntries; i++)
                {
                  image->colormap[i].red=ScaleCharToQuantum((char)
                    ReadBlobByte(image));
                  image->colormap[i].green=ScaleCharToQuantum((char)
                    ReadBlobByte(image));
                  image->colormap[i].blue=ScaleCharToQuantum((char)
                    ReadBlobByte(image));
                  (void) ReadBlobByte(image);   /*Opacity??*/
                }
              break;
            case 0x0E:
              Bitmap2Header1.Width=ReadBlobLSBShort(image);
              Bitmap2Header1.Height=ReadBlobLSBShort(image);
              if ((Bitmap2Header1.Width == 0) || (Bitmap2Header1.Height == 0))
                ThrowReaderException(CorruptImageError,"ImproperImageHeader");
              Bitmap2Header1.Depth=ReadBlobByte(image);
              Bitmap2Header1.Compression=ReadBlobByte(image);

              if(Bitmap2Header1.Compression > 1)
                continue; /*Unknown compression method */
              switch(Bitmap2Header1.Depth)
                {
                case 1:
                  bpp=1;
                  break;
                case 2:
                  bpp=2;
                  break;
                case 3:
                  bpp=4;
                  break;
                case 4:
                  bpp=8;
                  break;
                case 8:
                  bpp=24;
                  break;
                default:
                  continue;  /*Ignore raster with unknown depth*/
                }
              image->columns=Bitmap2Header1.Width;
              image->rows=Bitmap2Header1.Height;

              if ((image->colors == 0) && (bpp != 24))
                {
                  size_t
                    one;

                  one=1;
                  image->colors=one << bpp;
                  if (!AcquireImageColormap(image,image->colors))
                    goto NoMemory;
                }
              else
                {
                  if(bpp < 24)
                    if( image->colors<(one << bpp) && bpp!=24 )
                      image->colormap=(PixelPacket *) ResizeQuantumMemory(
                       image->colormap,(size_t) (one << bpp),
                       sizeof(*image->colormap));
                }


              switch(Bitmap2Header1.Compression)
                {
                case 0:    /*Uncompressed raster*/
                  {
                    ldblk=(ssize_t) ((bpp*image->columns+7)/8);
                    BImgBuff=(unsigned char *) AcquireQuantumMemory((size_t)
                      ldblk,sizeof(*BImgBuff));
                    if (BImgBuff == (unsigned char *) NULL)
                      goto NoMemory;

                    for(i=0; i< (ssize_t) image->rows; i++)
                      {
                        (void) ReadBlob(image,ldblk,BImgBuff);
                        InsertRow(BImgBuff,i,image,bpp);
                       }

                     if(BImgBuff)
                      BImgBuff=(unsigned char *) RelinquishMagickMemory(BImgBuff);;
                     break;
                   }
                 case 1:    /*RLE for WPG2 */
                  {
                    if( UnpackWPG2Raster(image,bpp) < 0)
                      goto DecompressionFailed;
                    break;
                  }
                }

              if(CTM[0][0]<0 && !image_info->ping)
                {    /*?? RotAngle=360-RotAngle;*/
                  Image
                    *flop_image;

                  flop_image = FlopImage(image, exception);
                  if (flop_image != (Image *) NULL) {
                    DuplicateBlob(flop_image,image);
                    (void) RemoveLastImageFromList(&image);
                    AppendImageToList(&image,flop_image);
                  }
                  /* Try to change CTM according to Flip - I am not sure, must be checked.
                     Tx(0,0)=-1;      Tx(1,0)=0;   Tx(2,0)=0;
                     Tx(0,1)= 0;      Tx(1,1)=1;   Tx(2,1)=0;
                     Tx(0,2)=(WPG._2Rect.X_ur+WPG._2Rect.X_ll);
                     Tx(1,2)=0;   Tx(2,2)=1; */
                }
              if(CTM[1][1]<0 && !image_info->ping)
                {    /*?? RotAngle=360-RotAngle;*/
                  Image
                    *flip_image;

                  flip_image = FlipImage(image, exception);
                  if (flip_image != (Image *) NULL) {
                    DuplicateBlob(flip_image,image);
                    (void) RemoveLastImageFromList(&image);
                    AppendImageToList(&image,flip_image);
                  }
                  /* Try to change CTM according to Flip - I am not sure, must be checked.
                     float_matrix Tx(3,3);
                     Tx(0,0)= 1;   Tx(1,0)= 0;   Tx(2,0)=0;
                     Tx(0,1)= 0;   Tx(1,1)=-1;   Tx(2,1)=0;
                     Tx(0,2)= 0;   Tx(1,2)=(WPG._2Rect.Y_ur+WPG._2Rect.Y_ll);
                     Tx(2,2)=1; */
                }


              /* Allocate next image structure. */
              AcquireNextImage(image_info,image);
              image->depth=8;
              if (image->next == (Image *) NULL)
                goto Finish;
              image=SyncNextImageInList(image);
              image->columns=image->rows=1;
              image->colors=0;
              break;

            case 0x12:  /* Postscript WPG2*/
        i=ReadBlobLSBShort(image);
              if(Rec2.RecordLength > (unsigned int) i)
                image=ExtractPostscript(image,image_info,
                  TellBlob(image)+i,    /*skip PS header in the wpg2*/
                  (ssize_t) (Rec2.RecordLength-i-2),exception);
              break;

      case 0x1B:          /*bitmap rectangle*/
              WPG2Flags = LoadWPG2Flags(image,StartWPG.PosSizePrecision,NULL,&CTM);
              (void) WPG2Flags;
              break;
            }
        }

      break;

    default:
      {
         ThrowReaderException(CoderError,"DataEncodingSchemeIsNotSupported");
      }
   }
  status=SetImageExtent(image,image->columns,image->rows);
  if (status == MagickFalse)
    {
      InheritException(exception,&image->exception);
      return(DestroyImageList(image));
    }

 Finish:
  (void) CloseBlob(image);

  {
    Image
      *p;

    ssize_t
      scene=0;

    /*
      Rewind list, removing any empty images while rewinding.
    */
    p=image;
    image=NULL;
    while (p != (Image *) NULL)
      {
        Image *tmp=p;
        if ((p->rows == 0) || (p->columns == 0)) {
          p=p->previous;
          DeleteImageFromList(&tmp);
        } else {
          image=p;
          p=p->previous;
        }
      }
    /*
      Fix scene numbers.
    */
    for (p=image; p != (Image *) NULL; p=p->next)
      p->scene=(size_t) scene++;
  }
  if (image == (Image *) NULL)
    ThrowReaderException(CorruptImageError,
      "ImageFileDoesNotContainAnyImageData");
  return(image);
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `bv_train_22472`
- **Project:** `ImageMagick`
- **Commit ID:** `a251039393f423c7858e63cab6aa98d17b8b7a41`
- **Tokens:** 11438

```c
static Image *ReadWPGImage(const ImageInfo *image_info,
  ExceptionInfo *exception)
{
  typedef struct
  {
    size_t FileId;
    MagickOffsetType DataOffset;
    unsigned int ProductType;
    unsigned int FileType;
    unsigned char MajorVersion;
    unsigned char MinorVersion;
    unsigned int EncryptKey;
    unsigned int Reserved;
  } WPGHeader;

  typedef struct
  {
    unsigned char RecType;
    size_t RecordLength;
  } WPGRecord;

  typedef struct
  {
    unsigned char Class;
    unsigned char RecType;
    size_t Extension;
    size_t RecordLength;
  } WPG2Record;

  typedef struct
  {
    unsigned  HorizontalUnits;
    unsigned  VerticalUnits;
    unsigned char PosSizePrecision;
  } WPG2Start;

  typedef struct
  {
    unsigned int Width;
    unsigned int Height;
    unsigned int Depth;
    unsigned int HorzRes;
    unsigned int VertRes;
  } WPGBitmapType1;

  typedef struct
  {
    unsigned int Width;
    unsigned int Height;
    unsigned char Depth;
    unsigned char Compression;
  } WPG2BitmapType1;

  typedef struct
  {
    unsigned int RotAngle;
    unsigned int LowLeftX;
    unsigned int LowLeftY;
    unsigned int UpRightX;
    unsigned int UpRightY;
    unsigned int Width;
    unsigned int Height;
    unsigned int Depth;
    unsigned int HorzRes;
    unsigned int VertRes;
  } WPGBitmapType2;

  typedef struct
  {
    unsigned int StartIndex;
    unsigned int NumOfEntries;
  } WPGColorMapRec;

  /*
  typedef struct {
    size_t PS_unknown1;
    unsigned int PS_unknown2;
    unsigned int PS_unknown3;
  } WPGPSl1Record;
  */

  Image
    *image;

  unsigned int
    status;

  WPGHeader
    Header;

  WPGRecord
    Rec;

  WPG2Record
    Rec2;

  WPG2Start StartWPG;

  WPGBitmapType1
    BitmapHeader1;

  WPG2BitmapType1
    Bitmap2Header1;

  WPGBitmapType2
    BitmapHeader2;

  WPGColorMapRec
    WPG_Palette;

  int
    i,
    bpp,
    WPG2Flags;

  ssize_t
    ldblk;

  size_t
    one;

  unsigned char
    *BImgBuff;

  tCTM CTM;         /*current transform matrix*/

  /*
    Open image file.
  */
  assert(image_info != (const ImageInfo *) NULL);
  assert(image_info->signature == MagickSignature);
  assert(exception != (ExceptionInfo *) NULL);
  assert(exception->signature == MagickSignature);
  one=1;
  image=AcquireImage(image_info);
  image->depth=8;
  status=OpenBlob(image_info,image,ReadBinaryBlobMode,exception);
  if (status == MagickFalse)
    {
      image=DestroyImageList(image);
      return((Image *) NULL);
    }
  /*
    Read WPG image.
  */
  Header.FileId=ReadBlobLSBLong(image);
  Header.DataOffset=(MagickOffsetType) ReadBlobLSBLong(image);
  Header.ProductType=ReadBlobLSBShort(image);
  Header.FileType=ReadBlobLSBShort(image);
  Header.MajorVersion=ReadBlobByte(image);
  Header.MinorVersion=ReadBlobByte(image);
  Header.EncryptKey=ReadBlobLSBShort(image);
  Header.Reserved=ReadBlobLSBShort(image);

  if (Header.FileId!=0x435057FF || (Header.ProductType>>8)!=0x16)
    ThrowReaderException(CorruptImageError,"ImproperImageHeader");
  if (Header.EncryptKey!=0)
    ThrowReaderException(CoderError,"EncryptedWPGImageFileNotSupported");

  image->columns = 1;
  image->rows = 1;
  image->colors = 0;
  bpp=0;
  BitmapHeader2.RotAngle=0;

  switch(Header.FileType)
    {
    case 1:     /* WPG level 1 */
      while(!EOFBlob(image)) /* object parser loop */
        {
          (void) SeekBlob(image,Header.DataOffset,SEEK_SET);
          if(EOFBlob(image))
            break;

          Rec.RecType=(i=ReadBlobByte(image));
          if(i==EOF)
            break;
          Rd_WP_DWORD(image,&Rec.RecordLength);
          if(EOFBlob(image))
            break;

          Header.DataOffset=TellBlob(image)+Rec.RecordLength;

          switch(Rec.RecType)
            {
            case 0x0B: /* bitmap type 1 */
              BitmapHeader1.Width=ReadBlobLSBShort(image);
              BitmapHeader1.Height=ReadBlobLSBShort(image);
              if ((BitmapHeader1.Width == 0) || (BitmapHeader1.Height == 0))
                ThrowReaderException(CorruptImageError,"ImproperImageHeader");
              BitmapHeader1.Depth=ReadBlobLSBShort(image);
              BitmapHeader1.HorzRes=ReadBlobLSBShort(image);
              BitmapHeader1.VertRes=ReadBlobLSBShort(image);

              if(BitmapHeader1.HorzRes && BitmapHeader1.VertRes)
                {
                  image->units=PixelsPerCentimeterResolution;
                  image->x_resolution=BitmapHeader1.HorzRes/470.0;
                  image->y_resolution=BitmapHeader1.VertRes/470.0;
                }
              image->columns=BitmapHeader1.Width;
              image->rows=BitmapHeader1.Height;
              bpp=BitmapHeader1.Depth;

              goto UnpackRaster;

            case 0x0E:  /*Color palette */
              WPG_Palette.StartIndex=ReadBlobLSBShort(image);
              WPG_Palette.NumOfEntries=ReadBlobLSBShort(image);

              image->colors=WPG_Palette.NumOfEntries;
              if (!AcquireImageColormap(image,image->colors))
                goto NoMemory;
              for (i=WPG_Palette.StartIndex;
                   i < (int)WPG_Palette.NumOfEntries; i++)
                {
                  image->colormap[i].red=ScaleCharToQuantum((unsigned char)
                    ReadBlobByte(image));
                  image->colormap[i].green=ScaleCharToQuantum((unsigned char)
                    ReadBlobByte(image));
                  image->colormap[i].blue=ScaleCharToQuantum((unsigned char)
                    ReadBlobByte(image));
                }
              break;

            case 0x11:  /* Start PS l1 */
              if(Rec.RecordLength > 8)
                image=ExtractPostscript(image,image_info,
                  TellBlob(image)+8,   /* skip PS header in the wpg */
                  (ssize_t) Rec.RecordLength-8,exception);
              break;

            case 0x14:  /* bitmap type 2 */
              BitmapHeader2.RotAngle=ReadBlobLSBShort(image);
              BitmapHeader2.LowLeftX=ReadBlobLSBShort(image);
              BitmapHeader2.LowLeftY=ReadBlobLSBShort(image);
              BitmapHeader2.UpRightX=ReadBlobLSBShort(image);
              BitmapHeader2.UpRightY=ReadBlobLSBShort(image);
              BitmapHeader2.Width=ReadBlobLSBShort(image);
              BitmapHeader2.Height=ReadBlobLSBShort(image);
              if ((BitmapHeader2.Width == 0) || (BitmapHeader2.Height == 0))
                ThrowReaderException(CorruptImageError,"ImproperImageHeader");
              BitmapHeader2.Depth=ReadBlobLSBShort(image);
              BitmapHeader2.HorzRes=ReadBlobLSBShort(image);
              BitmapHeader2.VertRes=ReadBlobLSBShort(image);

              image->units=PixelsPerCentimeterResolution;
              image->page.width=(unsigned int)
                ((BitmapHeader2.LowLeftX-BitmapHeader2.UpRightX)/470.0);
              image->page.height=(unsigned int)
                ((BitmapHeader2.LowLeftX-BitmapHeader2.UpRightY)/470.0);
              image->page.x=(int) (BitmapHeader2.LowLeftX/470.0);
              image->page.y=(int) (BitmapHeader2.LowLeftX/470.0);
              if(BitmapHeader2.HorzRes && BitmapHeader2.VertRes)
                {
                  image->x_resolution=BitmapHeader2.HorzRes/470.0;
                  image->y_resolution=BitmapHeader2.VertRes/470.0;
                }
              image->columns=BitmapHeader2.Width;
              image->rows=BitmapHeader2.Height;
              bpp=BitmapHeader2.Depth;

            UnpackRaster:
              if ((image->colors == 0) && (bpp != 24))
                {
                  image->colors=one << bpp;
                  if (!AcquireImageColormap(image,image->colors))
                    {
                    NoMemory:
                      ThrowReaderException(ResourceLimitError,
                        "MemoryAllocationFailed");
                    }
                  /* printf("Load default colormap \n"); */
                  for (i=0; (i < (int) image->colors) && (i < 256); i++)
                    {
                      image->colormap[i].red=ScaleCharToQuantum(WPG1_Palette[i].Red);
                      image->colormap[i].green=ScaleCharToQuantum(WPG1_Palette[i].Green);
                      image->colormap[i].blue=ScaleCharToQuantum(WPG1_Palette[i].Blue);
                    }
                }
              else
                {
                  if (bpp < 24)
                    if ( (image->colors < (one << bpp)) && (bpp != 24) )
                      image->colormap=(PixelPacket *) ResizeQuantumMemory(
                        image->colormap,(size_t) (one << bpp),
                        sizeof(*image->colormap));
                }

              if (bpp == 1)
                {
                  if(image->colormap[0].red==0 &&
                     image->colormap[0].green==0 &&
                     image->colormap[0].blue==0 &&
                     image->colormap[1].red==0 &&
                     image->colormap[1].green==0 &&
                     image->colormap[1].blue==0)
                    {  /* fix crippled monochrome palette */
                      image->colormap[1].red =
                        image->colormap[1].green =
                        image->colormap[1].blue = QuantumRange;
                    }
                }

              if(UnpackWPGRaster(image,bpp) < 0)
                /* The raster cannot be unpacked */
                {
                DecompressionFailed:
                  ThrowReaderException(CoderError,"UnableToDecompressImage");
                    }

              if(Rec.RecType==0x14 && BitmapHeader2.RotAngle!=0 && !image_info->ping)
                {
                  /* flop command */
                  if(BitmapHeader2.RotAngle & 0x8000)
                    {
                      Image
                        *flop_image;

                      flop_image = FlopImage(image, exception);
                      if (flop_image != (Image *) NULL) {
                        DuplicateBlob(flop_image,image);
                        (void) RemoveLastImageFromList(&image);
                        AppendImageToList(&image,flop_image);
                      }
                    }
                  /* flip command */
                  if(BitmapHeader2.RotAngle & 0x2000)
                    {
                      Image
                        *flip_image;

                      flip_image = FlipImage(image, exception);
                      if (flip_image != (Image *) NULL) {
                        DuplicateBlob(flip_image,image);
                        (void) RemoveLastImageFromList(&image);
                        AppendImageToList(&image,flip_image);
                      }
                    }

      /* rotate command */
                  if(BitmapHeader2.RotAngle & 0x0FFF)
                    {
                      Image
                        *rotate_image;

                      rotate_image=RotateImage(image,(BitmapHeader2.RotAngle &
                        0x0FFF), exception);
                      if (rotate_image != (Image *) NULL) {
                        DuplicateBlob(rotate_image,image);
                        (void) RemoveLastImageFromList(&image);
                        AppendImageToList(&image,rotate_image);
                      }
                    }
                }

              /* Allocate next image structure. */
              AcquireNextImage(image_info,image);
              image->depth=8;
              if (image->next == (Image *) NULL)
                goto Finish;
              image=SyncNextImageInList(image);
              image->columns=image->rows=0;
              image->colors=0;
              break;

            case 0x1B:  /* Postscript l2 */
              if(Rec.RecordLength>0x3C)
                image=ExtractPostscript(image,image_info,
                  TellBlob(image)+0x3C,   /* skip PS l2 header in the wpg */
                  (ssize_t) Rec.RecordLength-0x3C,exception);
              break;
            }
        }
      break;

    case 2:  /* WPG level 2 */
      (void) memset(CTM,0,sizeof(CTM));
      StartWPG.PosSizePrecision = 0;
      while(!EOFBlob(image)) /* object parser loop */
        {
          (void) SeekBlob(image,Header.DataOffset,SEEK_SET);
          if(EOFBlob(image))
            break;

          Rec2.Class=(i=ReadBlobByte(image));
          if(i==EOF)
            break;
          Rec2.RecType=(i=ReadBlobByte(image));
          if(i==EOF)
            break;
          Rd_WP_DWORD(image,&Rec2.Extension);
          Rd_WP_DWORD(image,&Rec2.RecordLength);
          if(EOFBlob(image))
            break;

          Header.DataOffset=TellBlob(image)+Rec2.RecordLength;

          switch(Rec2.RecType)
            {
      case 1:
              StartWPG.HorizontalUnits=ReadBlobLSBShort(image);
              StartWPG.VerticalUnits=ReadBlobLSBShort(image);
              StartWPG.PosSizePrecision=ReadBlobByte(image);
              break;
            case 0x0C:    /* Color palette */
              WPG_Palette.StartIndex=ReadBlobLSBShort(image);
              WPG_Palette.NumOfEntries=ReadBlobLSBShort(image);

              image->colors=WPG_Palette.NumOfEntries;
              if (AcquireImageColormap(image,image->colors) == MagickFalse)
                ThrowReaderException(ResourceLimitError,
                  "MemoryAllocationFailed");
              for (i=WPG_Palette.StartIndex;
                   i < (int)WPG_Palette.NumOfEntries; i++)
                {
                  image->colormap[i].red=ScaleCharToQuantum((char)
                    ReadBlobByte(image));
                  image->colormap[i].green=ScaleCharToQuantum((char)
                    ReadBlobByte(image));
                  image->colormap[i].blue=ScaleCharToQuantum((char)
                    ReadBlobByte(image));
                  (void) ReadBlobByte(image);   /*Opacity??*/
                }
              break;
            case 0x0E:
              Bitmap2Header1.Width=ReadBlobLSBShort(image);
              Bitmap2Header1.Height=ReadBlobLSBShort(image);
              if ((Bitmap2Header1.Width == 0) || (Bitmap2Header1.Height == 0))
                ThrowReaderException(CorruptImageError,"ImproperImageHeader");
              Bitmap2Header1.Depth=ReadBlobByte(image);
              Bitmap2Header1.Compression=ReadBlobByte(image);

              if(Bitmap2Header1.Compression > 1)
                continue; /*Unknown compression method */
              switch(Bitmap2Header1.Depth)
                {
                case 1:
                  bpp=1;
                  break;
                case 2:
                  bpp=2;
                  break;
                case 3:
                  bpp=4;
                  break;
                case 4:
                  bpp=8;
                  break;
                case 8:
                  bpp=24;
                  break;
                default:
                  continue;  /*Ignore raster with unknown depth*/
                }
              image->columns=Bitmap2Header1.Width;
              image->rows=Bitmap2Header1.Height;

              if ((image->colors == 0) && (bpp != 24))
                {
                  size_t
                    one;

                  one=1;
                  image->colors=one << bpp;
                  if (!AcquireImageColormap(image,image->colors))
                    goto NoMemory;
                }
              else
                {
                  if(bpp < 24)
                    if( image->colors<(one << bpp) && bpp!=24 )
                      image->colormap=(PixelPacket *) ResizeQuantumMemory(
                       image->colormap,(size_t) (one << bpp),
                       sizeof(*image->colormap));
                }


              switch(Bitmap2Header1.Compression)
                {
                case 0:    /*Uncompressed raster*/
                  {
                    ldblk=(ssize_t) ((bpp*image->columns+7)/8);
                    BImgBuff=(unsigned char *) AcquireQuantumMemory((size_t)
                      ldblk,sizeof(*BImgBuff));
                    if (BImgBuff == (unsigned char *) NULL)
                      goto NoMemory;

                    for(i=0; i< (ssize_t) image->rows; i++)
                      {
                        (void) ReadBlob(image,ldblk,BImgBuff);
                        InsertRow(BImgBuff,i,image,bpp);
                      }

                    if(BImgBuff)
                      BImgBuff=(unsigned char *) RelinquishMagickMemory(BImgBuff);;
                    break;
                  }
                case 1:    /*RLE for WPG2 */
                  {
                    if( UnpackWPG2Raster(image,bpp) < 0)
                      goto DecompressionFailed;
                    break;
                  }
                }

              if(CTM[0][0]<0 && !image_info->ping)
                {    /*?? RotAngle=360-RotAngle;*/
                  Image
                    *flop_image;

                  flop_image = FlopImage(image, exception);
                  if (flop_image != (Image *) NULL) {
                    DuplicateBlob(flop_image,image);
                    (void) RemoveLastImageFromList(&image);
                    AppendImageToList(&image,flop_image);
                  }
                  /* Try to change CTM according to Flip - I am not sure, must be checked.
                     Tx(0,0)=-1;      Tx(1,0)=0;   Tx(2,0)=0;
                     Tx(0,1)= 0;      Tx(1,1)=1;   Tx(2,1)=0;
                     Tx(0,2)=(WPG._2Rect.X_ur+WPG._2Rect.X_ll);
                     Tx(1,2)=0;   Tx(2,2)=1; */
                }
              if(CTM[1][1]<0 && !image_info->ping)
                {    /*?? RotAngle=360-RotAngle;*/
                  Image
                    *flip_image;

                  flip_image = FlipImage(image, exception);
                  if (flip_image != (Image *) NULL) {
                    DuplicateBlob(flip_image,image);
                    (void) RemoveLastImageFromList(&image);
                    AppendImageToList(&image,flip_image);
                  }
                  /* Try to change CTM according to Flip - I am not sure, must be checked.
                     float_matrix Tx(3,3);
                     Tx(0,0)= 1;   Tx(1,0)= 0;   Tx(2,0)=0;
                     Tx(0,1)= 0;   Tx(1,1)=-1;   Tx(2,1)=0;
                     Tx(0,2)= 0;   Tx(1,2)=(WPG._2Rect.Y_ur+WPG._2Rect.Y_ll);
                     Tx(2,2)=1; */
                }


              /* Allocate next image structure. */
              AcquireNextImage(image_info,image);
              image->depth=8;
              if (image->next == (Image *) NULL)
                goto Finish;
              image=SyncNextImageInList(image);
              image->columns=image->rows=1;
              image->colors=0;
              break;

            case 0x12:  /* Postscript WPG2*/
        i=ReadBlobLSBShort(image);
              if(Rec2.RecordLength > (unsigned int) i)
                image=ExtractPostscript(image,image_info,
                  TellBlob(image)+i,    /*skip PS header in the wpg2*/
                  (ssize_t) (Rec2.RecordLength-i-2),exception);
              break;

      case 0x1B:          /*bitmap rectangle*/
              WPG2Flags = LoadWPG2Flags(image,StartWPG.PosSizePrecision,NULL,&CTM);
              (void) WPG2Flags;
              break;
            }
        }

      break;

    default:
      {
         ThrowReaderException(CoderError,"DataEncodingSchemeIsNotSupported");
      }
   }
  status=SetImageExtent(image,image->columns,image->rows);
  if (status == MagickFalse)
    {
      InheritException(exception,&image->exception);
      return(DestroyImageList(image));
    }

 Finish:
  (void) CloseBlob(image);

  {
    Image
      *p;

    ssize_t
      scene=0;

    /*
      Rewind list, removing any empty images while rewinding.
    */
    p=image;
    image=NULL;
    while (p != (Image *) NULL)
      {
        Image *tmp=p;
        if ((p->rows == 0) || (p->columns == 0)) {
          p=p->previous;
          DeleteImageFromList(&tmp);
        } else {
          image=p;
          p=p->previous;
        }
      }
    /*
      Fix scene numbers.
    */
    for (p=image; p != (Image *) NULL; p=p->next)
      p->scene=(size_t) scene++;
  }
  if (image == (Image *) NULL)
    ThrowReaderException(CorruptImageError,
      "ImageFileDoesNotContainAnyImageData");
  return(image);
}
```

---

## Pair 5 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `dv_train_21385`
- **Project:** `gdk-pixbuf`
- **Commit ID:** `3bac204e0d0241a0d68586ece7099e6acf0e9bea`
- **CWE:** `CWE-20`
- **Tokens:** 52

```c
gdk_pixbuf__tiff_image_load (FILE *f, GError **error)
{
	return gdk_pixbuf__tiff_image_load_real (f, NULL, error);
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_214526`
- **Project:** `gdk-pixbuf`
- **Commit ID:** `4f0f465f991cd454d03189497f923eb40c170c22`
- **Tokens:** 62

```c
gdk_pixbuf__xbm_image_load (FILE    *f,
			    GError **error)
{
	return gdk_pixbuf__xbm_image_load_real (f, NULL, error);
}
```

---

## Pair 6 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `devign_train_3689`
- **Project:** `qemu`
- **Commit ID:** `0b8b8753e4d94901627b3e86431230f2319215c4`
- **CWE:** `None`
- **Tokens:** 66

```c
static void thread_pool_co_cb(void *opaque, int ret)

{

    ThreadPoolCo *co = opaque;



    co->ret = ret;

    qemu_coroutine_enter(co->co, NULL);

}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `devign_train_7024`
- **Project:** `qemu`
- **Commit ID:** `08844473820c93541fc47bdfeae0f2cc88cfab59`
- **Tokens:** 74

```c
static void bdrv_co_io_em_complete(void *opaque, int ret)

{

    CoroutineIOCompletion *co = opaque;



    co->ret = ret;

    qemu_coroutine_enter(co->coroutine, NULL);

}
```

---

## Pair 7 (Group Size: 3)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `devign_train_12914`
- **Project:** `qemu`
- **Commit ID:** `09aaa1602f9381c0e0fb539390b1793e51bdfc7b`
- **CWE:** `None`
- **Tokens:** 32

```c
void wdt_i6300esb_init(void)

{

    watchdog_add_model(&model);

}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_4229`
- **Project:** `net`
- **Commit ID:** `36d5fe6a000790f56039afe26834265db0a3ad4c`
- **Tokens:** 25

```c
void ovs_lock(void)
{
	mutex_lock(&ovs_mutex);
}
```

---

## Pair 8 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `dv_train_148666`
- **Project:** `systemd`
- **Commit ID:** `f5a8c43f39937d97c9ed75e3fe8621945b42b0db`
- **CWE:** `CWE-120`
- **Tokens:** 404

```c
static void ndisc_handler(sd_ndisc *nd, int event, void *userdata) {
        Link *link = userdata;
        int r;

        assert(link);

        if (IN_SET(link->state, LINK_STATE_FAILED, LINK_STATE_LINGER))
                return;

        switch (event) {
        case SD_NDISC_EVENT_TIMEOUT:
                dhcp6_request_address(link);

                r = sd_dhcp6_client_start(link->dhcp6_client);
                if (r < 0 && r != -EBUSY)
                        log_link_warning_errno(link, r, "Starting DHCPv6 client after NDisc timeout failed: %m");
                break;
        case SD_NDISC_EVENT_STOP:
                break;
        default:
                log_link_warning(link, "IPv6 Neighbor Discovery unknown event: %d", event);
        }
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_934`
- **Project:** `systemd`
- **Commit ID:** `f5a8c43f39937d97c9ed75e3fe8621945b42b0db`
- **Tokens:** 405

```c
static void ndisc_handler(sd_ndisc *nd, int event, void *userdata) {
        Link *link = userdata;
        int r;

        assert(link);

        if (IN_SET(link->state, LINK_STATE_FAILED, LINK_STATE_LINGER))
                return;

        switch (event) {
        case SD_NDISC_EVENT_TIMEOUT:
                dhcp6_request_address(link);

                r = sd_dhcp6_client_start(link->dhcp6_client);
                if (r < 0 && r != -EALREADY)
                        log_link_warning_errno(link, r, "Starting DHCPv6 client after NDisc timeout failed: %m");
                break;
        case SD_NDISC_EVENT_STOP:
                break;
        default:
                log_link_warning(link, "IPv6 Neighbor Discovery unknown event: %d", event);
        }
}
```

---

## Pair 9 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `dv_train_110491`
- **Project:** `libvpx`
- **Commit ID:** `0681cff1ad36b3ef8ec242f59b5a6c4234ccfb88`
- **CWE:** `CWE-125`
- **Tokens:** 1207

```c
static vpx_codec_err_t decoder_peek_si_internal(
    const uint8_t *data, unsigned int data_sz, vpx_codec_stream_info_t *si,
    int *is_intra_only, vpx_decrypt_cb decrypt_cb, void *decrypt_state) {
  int intra_only_flag = 0;
  uint8_t clear_buffer[10];

  if (data + data_sz <= data) return VPX_CODEC_INVALID_PARAM;

  si->is_kf = 0;
  si->w = si->h = 0;

  if (decrypt_cb) {
    data_sz = VPXMIN(sizeof(clear_buffer), data_sz);
    decrypt_cb(decrypt_state, data, clear_buffer, data_sz);
    data = clear_buffer;
  }

  // A maximum of 6 bits are needed to read the frame marker, profile and
  // show_existing_frame.
  if (data_sz < 1) return VPX_CODEC_UNSUP_BITSTREAM;

  {
    int show_frame;
    int error_resilient;
    struct vpx_read_bit_buffer rb = { data, data + data_sz, 0, NULL, NULL };
    const int frame_marker = vpx_rb_read_literal(&rb, 2);
    const BITSTREAM_PROFILE profile = vp9_read_profile(&rb);

    if (frame_marker != VP9_FRAME_MARKER) return VPX_CODEC_UNSUP_BITSTREAM;

    if (profile >= MAX_PROFILES) return VPX_CODEC_UNSUP_BITSTREAM;

    if (vpx_rb_read_bit(&rb)) {  // show an existing frame
      // If profile is > 2 and show_existing_frame is true, then at least 1 more
      // byte (6+3=9 bits) is needed.
      if (profile > 2 && data_sz < 2) return VPX_CODEC_UNSUP_BITSTREAM;
      vpx_rb_read_literal(&rb, 3);  // Frame buffer to show.
      return VPX_CODEC_OK;
    }

    // For the rest of the function, a maximum of 9 more bytes are needed
    // (computed by taking the maximum possible bits needed in each case). Note
    // that this has to be updated if we read any more bits in this function.
    if (data_sz < 10) return VPX_CODEC_UNSUP_BITSTREAM;

    si->is_kf = !vpx_rb_read_bit(&rb);
    show_frame = vpx_rb_read_bit(&rb);
    error_resilient = vpx_rb_read_bit(&rb);

    if (si->is_kf) {
      if (!vp9_read_sync_code(&rb)) return VPX_CODEC_UNSUP_BITSTREAM;

      if (!parse_bitdepth_colorspace_sampling(profile, &rb))
        return VPX_CODEC_UNSUP_BITSTREAM;
      vp9_read_frame_size(&rb, (int *)&si->w, (int *)&si->h);
    } else {
      intra_only_flag = show_frame ? 0 : vpx_rb_read_bit(&rb);

      rb.bit_offset += error_resilient ? 0 : 2;  // reset_frame_context

      if (intra_only_flag) {
        if (!vp9_read_sync_code(&rb)) return VPX_CODEC_UNSUP_BITSTREAM;
        if (profile > PROFILE_0) {
          if (!parse_bitdepth_colorspace_sampling(profile, &rb))
            return VPX_CODEC_UNSUP_BITSTREAM;
        }
        rb.bit_offset += REF_FRAMES;  // refresh_frame_flags
        vp9_read_frame_size(&rb, (int *)&si->w, (int *)&si->h);
      }
    }
  }
  if (is_intra_only != NULL) *is_intra_only = intra_only_flag;
  return VPX_CODEC_OK;
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `primevul_train_train:8063_fixed`
- **Project:** `Android`
- **Commit ID:** `4974dcbd0289a2530df2ee2a25b5f92775df80da`
- **Tokens:** 1254

```c
static vpx_codec_err_t decoder_peek_si_internal(const uint8_t *data,
 unsigned int data_sz,
 vpx_codec_stream_info_t *si,
 int *is_intra_only,

                                                 vpx_decrypt_cb decrypt_cb,
                                                 void *decrypt_state) {
   int intra_only_flag = 0;
  uint8_t clear_buffer[10];

   if (data + data_sz <= data)
     return VPX_CODEC_INVALID_PARAM;

  si->is_kf = 0;
  si->w = si->h = 0;

 if (decrypt_cb) {
    data_sz = VPXMIN(sizeof(clear_buffer), data_sz);
    decrypt_cb(decrypt_state, data, clear_buffer, data_sz);

     data = clear_buffer;
   }

  // A maximum of 6 bits are needed to read the frame marker, profile and
  // show_existing_frame.
  if (data_sz < 1)
    return VPX_CODEC_UNSUP_BITSTREAM;

   {
     int show_frame;
     int error_resilient;
 struct vpx_read_bit_buffer rb = { data, data + data_sz, 0, NULL, NULL };
 const int frame_marker = vpx_rb_read_literal(&rb, 2);
 const BITSTREAM_PROFILE profile = vp9_read_profile(&rb);

 if (frame_marker != VP9_FRAME_MARKER)
 return VPX_CODEC_UNSUP_BITSTREAM;


     if (profile >= MAX_PROFILES)
       return VPX_CODEC_UNSUP_BITSTREAM;

     if (vpx_rb_read_bit(&rb)) {  // show an existing frame
      // If profile is > 2 and show_existing_frame is true, then at least 1 more
      // byte (6+3=9 bits) is needed.
      if (profile > 2 && data_sz < 2)
        return VPX_CODEC_UNSUP_BITSTREAM;
       vpx_rb_read_literal(&rb, 3);  // Frame buffer to show.
       return VPX_CODEC_OK;
     }

    // For the rest of the function, a maximum of 9 more bytes are needed
    // (computed by taking the maximum possible bits needed in each case). Note
    // that this has to be updated if we read any more bits in this function.
    if (data_sz < 10)
       return VPX_CODEC_UNSUP_BITSTREAM;

     si->is_kf = !vpx_rb_read_bit(&rb);
    show_frame = vpx_rb_read_bit(&rb);
    error_resilient = vpx_rb_read_bit(&rb);

 if (si->is_kf) {
 if (!vp9_read_sync_code(&rb))
 return VPX_CODEC_UNSUP_BITSTREAM;

 if (!parse_bitdepth_colorspace_sampling(profile, &rb))
 return VPX_CODEC_UNSUP_BITSTREAM;
      vp9_read_frame_size(&rb, (int *)&si->w, (int *)&si->h);
 } else {
      intra_only_flag = show_frame ? 0 : vpx_rb_read_bit(&rb);

      rb.bit_offset += error_resilient ? 0 : 2; // reset_frame_context

 if (intra_only_flag) {
 if (!vp9_read_sync_code(&rb))
 return VPX_CODEC_UNSUP_BITSTREAM;
 if (profile > PROFILE_0) {
 if (!parse_bitdepth_colorspace_sampling(profile, &rb))
 return VPX_CODEC_UNSUP_BITSTREAM;
 }
        rb.bit_offset += REF_FRAMES; // refresh_frame_flags
        vp9_read_frame_size(&rb, (int *)&si->w, (int *)&si->h);
 }
 }
 }
 if (is_intra_only != NULL)
 *is_intra_only = intra_only_flag;
 return VPX_CODEC_OK;
}
```

---

## Pair 10 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `dv_train_70861`
- **Project:** `clamav-devel`
- **Commit ID:** `167c0079292814ec5523d0b97a9e1b002bf8819b`
- **CWE:** `None`
- **Tokens:** 226

```c
static int intermediates_eval(cli_ctx *ctx, struct cli_ac_lsig *ac_lsig)
{
    uint32_t i, icnt = ac_lsig->tdb.intermediates[0];
    int32_t j = -1;

    if (ctx->recursion < icnt)
        return 0;

    for (i = icnt; i > 0; i--) {
        if (ac_lsig->tdb.intermediates[i] == CL_TYPE_ANY)
            continue;
        if (ac_lsig->tdb.intermediates[i] != cli_get_container_type(ctx, j--))
            return 0;
    }
    return 1;
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_145796`
- **Project:** `clamav-devel`
- **Commit ID:** `167c0079292814ec5523d0b97a9e1b002bf8819b`
- **Tokens:** 227

```c
static int intermediates_eval(cli_ctx *ctx, struct cli_ac_lsig *ac_lsig)
{
    uint32_t i, icnt = ac_lsig->tdb.intermediates[0];
    int32_t j = -1;

    if (ctx->recursion < icnt)
        return 0;

    for (i = icnt; i > 0; i--) {
        if (ac_lsig->tdb.intermediates[i] == CL_TYPE_ANY)
            continue;
        if (ac_lsig->tdb.intermediates[i] != cli_get_container_intermediate(ctx, j--))
            return 0;
    }
    return 1;
}
```

---

## Pair 11 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `primevul_train_train:10322_vuln`
- **Project:** `qemu`
- **Commit ID:** `670e56d3ed2918b3861d9216f2c0540d9e9ae0d5`
- **CWE:** `None`
- **Tokens:** 1562

```c
static int mptsas_process_scsi_io_request(MPTSASState *s,
                                          MPIMsgSCSIIORequest *scsi_io,
                                          hwaddr addr)
{
    MPTSASRequest *req;
    MPIMsgSCSIIOReply reply;
    SCSIDevice *sdev;
    int status;

    mptsas_fix_scsi_io_endianness(scsi_io);

    trace_mptsas_process_scsi_io_request(s, scsi_io->Bus, scsi_io->TargetID,
                                         scsi_io->LUN[1], scsi_io->DataLength);

    status = mptsas_scsi_device_find(s, scsi_io->Bus, scsi_io->TargetID,
                                     scsi_io->LUN, &sdev);
    if (status) {
         goto bad;
     }

    req = g_new(MPTSASRequest, 1);
     QTAILQ_INSERT_TAIL(&s->pending, req, next);
     req->scsi_io = *scsi_io;
     req->dev = s;

    status = mptsas_build_sgl(s, req, addr);
    if (status) {
        goto free_bad;
    }

    if (req->qsg.size < scsi_io->DataLength) {
        trace_mptsas_sgl_overflow(s, scsi_io->MsgContext, scsi_io->DataLength,
                                  req->qsg.size);
        status = MPI_IOCSTATUS_INVALID_SGL;
        goto free_bad;
    }

    req->sreq = scsi_req_new(sdev, scsi_io->MsgContext,
                            scsi_io->LUN[1], scsi_io->CDB, req);

    if (req->sreq->cmd.xfer > scsi_io->DataLength) {
        goto overrun;
    }
    switch (scsi_io->Control & MPI_SCSIIO_CONTROL_DATADIRECTION_MASK) {
    case MPI_SCSIIO_CONTROL_NODATATRANSFER:
        if (req->sreq->cmd.mode != SCSI_XFER_NONE) {
            goto overrun;
        }
        break;

    case MPI_SCSIIO_CONTROL_WRITE:
        if (req->sreq->cmd.mode != SCSI_XFER_TO_DEV) {
            goto overrun;
        }
        break;

    case MPI_SCSIIO_CONTROL_READ:
        if (req->sreq->cmd.mode != SCSI_XFER_FROM_DEV) {
            goto overrun;
        }
        break;
    }

    if (scsi_req_enqueue(req->sreq)) {
        scsi_req_continue(req->sreq);
    }
    return 0;

overrun:
    trace_mptsas_scsi_overflow(s, scsi_io->MsgContext, req->sreq->cmd.xfer,
                               scsi_io->DataLength);
    status = MPI_IOCSTATUS_SCSI_DATA_OVERRUN;
free_bad:
    mptsas_free_request(req);
bad:
    memset(&reply, 0, sizeof(reply));
    reply.TargetID          = scsi_io->TargetID;
    reply.Bus               = scsi_io->Bus;
    reply.MsgLength         = sizeof(reply) / 4;
    reply.Function          = scsi_io->Function;
    reply.CDBLength         = scsi_io->CDBLength;
    reply.SenseBufferLength = scsi_io->SenseBufferLength;
    reply.MsgContext        = scsi_io->MsgContext;
    reply.SCSIState         = MPI_SCSI_STATE_NO_SCSI_STATUS;
    reply.IOCStatus         = status;

    mptsas_fix_scsi_io_reply_endianness(&reply);
    mptsas_reply(s, (MPIDefaultReply *)&reply);

    return 0;
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `devign_train_10195`
- **Project:** `qemu`
- **Commit ID:** `670e56d3ed2918b3861d9216f2c0540d9e9ae0d5`
- **Tokens:** 1575

```c
static int mptsas_process_scsi_io_request(MPTSASState *s,

                                          MPIMsgSCSIIORequest *scsi_io,

                                          hwaddr addr)

{

    MPTSASRequest *req;

    MPIMsgSCSIIOReply reply;

    SCSIDevice *sdev;

    int status;



    mptsas_fix_scsi_io_endianness(scsi_io);



    trace_mptsas_process_scsi_io_request(s, scsi_io->Bus, scsi_io->TargetID,

                                         scsi_io->LUN[1], scsi_io->DataLength);



    status = mptsas_scsi_device_find(s, scsi_io->Bus, scsi_io->TargetID,

                                     scsi_io->LUN, &sdev);

    if (status) {

        goto bad;

    }



    req = g_new(MPTSASRequest, 1);

    QTAILQ_INSERT_TAIL(&s->pending, req, next);

    req->scsi_io = *scsi_io;

    req->dev = s;



    status = mptsas_build_sgl(s, req, addr);

    if (status) {

        goto free_bad;

    }



    if (req->qsg.size < scsi_io->DataLength) {

        trace_mptsas_sgl_overflow(s, scsi_io->MsgContext, scsi_io->DataLength,

                                  req->qsg.size);

        status = MPI_IOCSTATUS_INVALID_SGL;

        goto free_bad;

    }



    req->sreq = scsi_req_new(sdev, scsi_io->MsgContext,

                            scsi_io->LUN[1], scsi_io->CDB, req);



    if (req->sreq->cmd.xfer > scsi_io->DataLength) {

        goto overrun;

    }

    switch (scsi_io->Control & MPI_SCSIIO_CONTROL_DATADIRECTION_MASK) {

    case MPI_SCSIIO_CONTROL_NODATATRANSFER:

        if (req->sreq->cmd.mode != SCSI_XFER_NONE) {

            goto overrun;

        }

        break;



    case MPI_SCSIIO_CONTROL_WRITE:

        if (req->sreq->cmd.mode != SCSI_XFER_TO_DEV) {

            goto overrun;

        }

        break;



    case MPI_SCSIIO_CONTROL_READ:

        if (req->sreq->cmd.mode != SCSI_XFER_FROM_DEV) {

            goto overrun;

        }

        break;

    }



    if (scsi_req_enqueue(req->sreq)) {

        scsi_req_continue(req->sreq);

    }

    return 0;



overrun:

    trace_mptsas_scsi_overflow(s, scsi_io->MsgContext, req->sreq->cmd.xfer,

                               scsi_io->DataLength);

    status = MPI_IOCSTATUS_SCSI_DATA_OVERRUN;

free_bad:

    mptsas_free_request(req);

bad:

    memset(&reply, 0, sizeof(reply));

    reply.TargetID          = scsi_io->TargetID;

    reply.Bus               = scsi_io->Bus;

    reply.MsgLength         = sizeof(reply) / 4;

    reply.Function          = scsi_io->Function;

    reply.CDBLength         = scsi_io->CDBLength;

    reply.SenseBufferLength = scsi_io->SenseBufferLength;

    reply.MsgContext        = scsi_io->MsgContext;

    reply.SCSIState         = MPI_SCSI_STATE_NO_SCSI_STATUS;

    reply.IOCStatus         = status;



    mptsas_fix_scsi_io_reply_endianness(&reply);

    mptsas_reply(s, (MPIDefaultReply *)&reply);



    return 0;

}
```

---

## Pair 12 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `dv_train_229913`
- **Project:** `mod_h2`
- **Commit ID:** `5e75e5685dd043fe93a5a08a15edd087a43f6968`
- **CWE:** `None`
- **Tokens:** 173

```c
static void h2_session_ev_data_read(h2_session *session, int arg, const char *msg)
{
    switch (session->state) {
        case H2_SESSION_ST_IDLE:
        case H2_SESSION_ST_WAIT:
            transit(session, "data read", H2_SESSION_ST_BUSY);
            break;
        default:
            /* nop */
            break;
    }
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_57890`
- **Project:** `mod_h2`
- **Commit ID:** `dd05d49abe0f67512ce9ed5ba422d7711effecfb`
- **Tokens:** 208

```c
static void ev_stream_submitted(h2_proxy_session *session, int stream_id,
                                const char *msg)
{
    switch (session->state) {
        case H2_PROXYS_ST_IDLE:
        case H2_PROXYS_ST_WAIT:
            transit(session, "stream submitted", H2_PROXYS_ST_BUSY);
            break;
        default:
            /* nop */
            break;
    }
}
```

---

## Pair 13 (Group Size: 7)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `bv_train_126722`
- **Project:** `Android`
- **Commit ID:** `04839626ed859623901ebd3a5fd483982186b59d`
- **CWE:** `CWE-119`
- **Tokens:** 26

```c
const SegmentInfo* Segment::GetInfo() const
{
    return m_pInfo;
 }
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_156612`
- **Project:** `libvpx`
- **Commit ID:** `34d54b04e98dd0bac32e9aab0fbda0bf501bc742`
- **Tokens:** 20

```c
const SegmentInfo* Segment::GetInfo() const { return m_pInfo; }
```

---

## Pair 14 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `dv_train_95584`
- **Project:** `jasper`
- **Commit ID:** `d42b2388f7f8e0332c846675133acea151fc557a`
- **CWE:** `CWE-20`
- **Tokens:** 111

```c
static int jas_iccputuint(jas_stream_t *out, int n, ulonglong val)
{
	int i;
	int c;
	for (i = n; i > 0; --i) {
		c = (val >> (8 * (i - 1))) & 0xff;
		if (jas_stream_putc(out, c) == EOF)
			return -1;
	}
	return 0;
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_101242`
- **Project:** `jasper`
- **Commit ID:** `d42b2388f7f8e0332c846675133acea151fc557a`
- **Tokens:** 114

```c
static int jas_iccputuint(jas_stream_t *out, int n, jas_ulonglong val)
{
	int i;
	int c;
	for (i = n; i > 0; --i) {
		c = (val >> (8 * (i - 1))) & 0xff;
		if (jas_stream_putc(out, c) == EOF)
			return -1;
	}
	return 0;
}
```

---

## Pair 15 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `dv_train_107008`
- **Project:** `jasper`
- **Commit ID:** `65536647d380571d1a9a6c91fa03775fb5bbd256`
- **CWE:** `CWE-119`
- **Tokens:** 619

```c
static jas_image_cmpt_t *jas_image_cmpt_create(uint_fast32_t tlx,
  uint_fast32_t tly, uint_fast32_t hstep, uint_fast32_t vstep,
  uint_fast32_t width, uint_fast32_t height, uint_fast16_t depth, bool sgnd,
  uint_fast32_t inmem)
{
	jas_image_cmpt_t *cmpt;
	size_t size;

	if (!(cmpt = jas_malloc(sizeof(jas_image_cmpt_t)))) {
		goto error;
	}

	cmpt->type_ = JAS_IMAGE_CT_UNKNOWN;
	cmpt->tlx_ = tlx;
	cmpt->tly_ = tly;
	cmpt->hstep_ = hstep;
	cmpt->vstep_ = vstep;
	cmpt->width_ = width;
	cmpt->height_ = height;
	cmpt->prec_ = depth;
	cmpt->sgnd_ = sgnd;
	cmpt->stream_ = 0;
	cmpt->cps_ = (depth + 7) / 8;

	// size = cmpt->width_ * cmpt->height_ * cmpt->cps_;
	if (!jas_safe_size_mul(cmpt->width_, cmpt->height_, &size) ||
	  !jas_safe_size_mul(size, cmpt->cps_, &size)) {
		goto error;
	}
	cmpt->stream_ = (inmem) ? jas_stream_memopen(0, size) :
	  jas_stream_tmpfile();
	if (!cmpt->stream_) {
		goto error;
	}

	/* Zero the component data.  This isn't necessary, but it is
	convenient for debugging purposes. */
	/* Note: conversion of size - 1 to long can overflow */
	if (jas_stream_seek(cmpt->stream_, size - 1, SEEK_SET) < 0 ||
	  jas_stream_putc(cmpt->stream_, 0) == EOF ||
	  jas_stream_seek(cmpt->stream_, 0, SEEK_SET) < 0) {
		goto error;
	}

	return cmpt;

error:
	if (cmpt) {
		jas_image_cmpt_destroy(cmpt);
	}
	return 0;
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `bv_train_30363`
- **Project:** `jasper`
- **Commit ID:** `dee11ec440d7908d1daf69f40a3324b27cf213ba`
- **Tokens:** 595

```c
static jas_image_cmpt_t *jas_image_cmpt_create(uint_fast32_t tlx,
  uint_fast32_t tly, uint_fast32_t hstep, uint_fast32_t vstep,
  uint_fast32_t width, uint_fast32_t height, uint_fast16_t depth, bool sgnd,
  uint_fast32_t inmem)
{
	jas_image_cmpt_t *cmpt;
	size_t size;

	if (!(cmpt = jas_malloc(sizeof(jas_image_cmpt_t)))) {
		goto error;
	}

	cmpt->type_ = JAS_IMAGE_CT_UNKNOWN;
	cmpt->tlx_ = tlx;
	cmpt->tly_ = tly;
	cmpt->hstep_ = hstep;
	cmpt->vstep_ = vstep;
	cmpt->width_ = width;
	cmpt->height_ = height;
	cmpt->prec_ = depth;
	cmpt->sgnd_ = sgnd;
	cmpt->stream_ = 0;
	cmpt->cps_ = (depth + 7) / 8;

	if (!jas_safe_size_mul(cmpt->width_, cmpt->height_, &size) ||
	  !jas_safe_size_mul(size, cmpt->cps_, &size)) {
		goto error;
	}
	cmpt->stream_ = (inmem) ? jas_stream_memopen(0, size) :
	  jas_stream_tmpfile();
	if (!cmpt->stream_) {
		goto error;
	}

	/* Zero the component data.  This isn't necessary, but it is
	convenient for debugging purposes. */
	/* Note: conversion of size - 1 to long can overflow */
	if (jas_stream_seek(cmpt->stream_, size - 1, SEEK_SET) < 0 ||
	  jas_stream_putc(cmpt->stream_, 0) == EOF ||
	  jas_stream_seek(cmpt->stream_, 0, SEEK_SET) < 0) {
		goto error;
	}

	return cmpt;

error:
	if (cmpt) {
		jas_image_cmpt_destroy(cmpt);
	}
	return 0;
}
```

---

## Pair 16 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `dv_train_77409`
- **Project:** `psutil`
- **Commit ID:** `7d512c8e4442a896d56505be3e78f1156f443465`
- **CWE:** `CWE-415`
- **Tokens:** 937

```c
psutil_users(PyObject *self, PyObject *args) {
    struct utmpx *ut;
    PyObject *py_retlist = PyList_New(0);
    PyObject *py_tuple = NULL;
    PyObject *py_username = NULL;
    PyObject *py_tty = NULL;
    PyObject *py_hostname = NULL;
    PyObject *py_user_proc = NULL;

    if (py_retlist == NULL)
        return NULL;

    setutxent();
    while (NULL != (ut = getutxent())) {
        if (ut->ut_type == USER_PROCESS)
            py_user_proc = Py_True;
        else
            py_user_proc = Py_False;
        py_username = PyUnicode_DecodeFSDefault(ut->ut_user);
        if (! py_username)
            goto error;
        py_tty = PyUnicode_DecodeFSDefault(ut->ut_line);
        if (! py_tty)
            goto error;
        py_hostname = PyUnicode_DecodeFSDefault(ut->ut_host);
        if (! py_hostname)
            goto error;
        py_tuple = Py_BuildValue(
            "(OOOfOi)",
            py_username,              // username
            py_tty,                   // tty
            py_hostname,              // hostname
            (float)ut->ut_tv.tv_sec,  // tstamp
            py_user_proc,             // (bool) user process
            ut->ut_pid                // process id
        );
        if (py_tuple == NULL)
            goto error;
        if (PyList_Append(py_retlist, py_tuple))
            goto error;
        Py_DECREF(py_username);
        Py_DECREF(py_tty);
        Py_DECREF(py_hostname);
        Py_DECREF(py_tuple);
    }
    endutxent();

    return py_retlist;

error:
    Py_XDECREF(py_username);
    Py_XDECREF(py_tty);
    Py_XDECREF(py_hostname);
    Py_XDECREF(py_tuple);
    Py_DECREF(py_retlist);
    if (ut != NULL)
        endutxent();
    return NULL;
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_2172`
- **Project:** `psutil`
- **Commit ID:** `7d512c8e4442a896d56505be3e78f1156f443465`
- **Tokens:** 937

```c
psutil_users(PyObject *self, PyObject *args) {
    struct utmpx *ut;
    PyObject *py_retlist = PyList_New(0);
    PyObject *py_tuple = NULL;
    PyObject *py_username = NULL;
    PyObject *py_tty = NULL;
    PyObject *py_hostname = NULL;
    PyObject *py_user_proc = NULL;

    if (py_retlist == NULL)
        return NULL;

    setutxent();
    while (NULL != (ut = getutxent())) {
        if (ut->ut_type == USER_PROCESS)
            py_user_proc = Py_True;
        else
            py_user_proc = Py_False;
        py_username = PyUnicode_DecodeFSDefault(ut->ut_user);
        if (! py_username)
            goto error;
        py_tty = PyUnicode_DecodeFSDefault(ut->ut_line);
        if (! py_tty)
            goto error;
        py_hostname = PyUnicode_DecodeFSDefault(ut->ut_host);
        if (! py_hostname)
            goto error;
        py_tuple = Py_BuildValue(
            "(OOOfOi)",
            py_username,              // username
            py_tty,                   // tty
            py_hostname,              // hostname
            (float)ut->ut_tv.tv_sec,  // tstamp
            py_user_proc,             // (bool) user process
            ut->ut_pid                // process id
        );
        if (py_tuple == NULL)
            goto error;
        if (PyList_Append(py_retlist, py_tuple))
            goto error;
        Py_CLEAR(py_username);
        Py_CLEAR(py_tty);
        Py_CLEAR(py_hostname);
        Py_CLEAR(py_tuple);
    }
    endutxent();

    return py_retlist;

error:
    Py_XDECREF(py_username);
    Py_XDECREF(py_tty);
    Py_XDECREF(py_hostname);
    Py_XDECREF(py_tuple);
    Py_DECREF(py_retlist);
    if (ut != NULL)
        endutxent();
    return NULL;
}
```

---

## Pair 17 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `dv_train_221592`
- **Project:** `libinput`
- **Commit ID:** `a423d7d3269dc32a87384f79e29bb5ac021c83d1`
- **CWE:** `CWE-134`
- **Tokens:** 756

```c
litest_utils_suite(void)
{
	TCase *tc;
	Suite *s;

	s = suite_create("litest:utils");
	tc = tcase_create("utils");

	tcase_add_test(tc, array_for_each);

	tcase_add_test(tc, bitfield_helpers);
	tcase_add_test(tc, matrix_helpers);
	tcase_add_test(tc, ratelimit_helpers);
	tcase_add_test(tc, dpi_parser);
	tcase_add_test(tc, wheel_click_parser);
	tcase_add_test(tc, wheel_click_count_parser);
	tcase_add_test(tc, dimension_prop_parser);
	tcase_add_test(tc, reliability_prop_parser);
	tcase_add_test(tc, calibration_prop_parser);
	tcase_add_test(tc, range_prop_parser);
	tcase_add_test(tc, boolean_prop_parser);
	tcase_add_test(tc, evcode_prop_parser);
	tcase_add_test(tc, input_prop_parser);
	tcase_add_test(tc, evdev_abs_parser);
	tcase_add_test(tc, safe_atoi_test);
	tcase_add_test(tc, safe_atoi_base_16_test);
	tcase_add_test(tc, safe_atoi_base_8_test);
	tcase_add_test(tc, safe_atou_test);
	tcase_add_test(tc, safe_atou_base_16_test);
	tcase_add_test(tc, safe_atou_base_8_test);
	tcase_add_test(tc, safe_atod_test);
	tcase_add_test(tc, strsplit_test);
	tcase_add_test(tc, strargv_test);
	tcase_add_test(tc, kvsplit_double_test);
	tcase_add_test(tc, strjoin_test);
	tcase_add_test(tc, strstrip_test);
	tcase_add_test(tc, strendswith_test);
	tcase_add_test(tc, strstartswith_test);
	tcase_add_test(tc, time_conversion);
	tcase_add_test(tc, human_time);

	tcase_add_test(tc, list_test_insert);
	tcase_add_test(tc, list_test_append);
	tcase_add_test(tc, list_test_foreach);
	tcase_add_test(tc, strverscmp_test);
	tcase_add_test(tc, streq_test);
	tcase_add_test(tc, strneq_test);
	tcase_add_test(tc, trunkname_test);
	tcase_add_test(tc, basename_test);

	suite_add_tcase(s, tc);

	return s;
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_74533`
- **Project:** `libinput`
- **Commit ID:** `04f22107e1a2ead05401d9169fa4306e8c7eefad`
- **Tokens:** 757

```c
litest_utils_suite(void)
{
	TCase *tc;
	Suite *s;

	s = suite_create("litest:utils");
	tc = tcase_create("utils");

	tcase_add_test(tc, bitfield_helpers);
	tcase_add_test(tc, matrix_helpers);
	tcase_add_test(tc, ratelimit_helpers);
	tcase_add_test(tc, dpi_parser);
	tcase_add_test(tc, wheel_click_parser);
	tcase_add_test(tc, wheel_click_count_parser);
	tcase_add_test(tc, dimension_prop_parser);
	tcase_add_test(tc, reliability_prop_parser);
	tcase_add_test(tc, calibration_prop_parser);
	tcase_add_test(tc, range_prop_parser);
	tcase_add_test(tc, boolean_prop_parser);
	tcase_add_test(tc, evcode_prop_parser);
	tcase_add_test(tc, input_prop_parser);
	tcase_add_test(tc, evdev_abs_parser);
	tcase_add_test(tc, safe_atoi_test);
	tcase_add_test(tc, safe_atoi_base_16_test);
	tcase_add_test(tc, safe_atoi_base_8_test);
	tcase_add_test(tc, safe_atou_test);
	tcase_add_test(tc, safe_atou_base_16_test);
	tcase_add_test(tc, safe_atou_base_8_test);
	tcase_add_test(tc, safe_atod_test);
	tcase_add_test(tc, strsplit_test);
	tcase_add_test(tc, strargv_test);
	tcase_add_test(tc, kvsplit_double_test);
	tcase_add_test(tc, strjoin_test);
	tcase_add_test(tc, strstrip_test);
	tcase_add_test(tc, strendswith_test);
	tcase_add_test(tc, strstartswith_test);
	tcase_add_test(tc, strsanitize_test);
	tcase_add_test(tc, time_conversion);
	tcase_add_test(tc, human_time);

	tcase_add_test(tc, list_test_insert);
	tcase_add_test(tc, list_test_append);
	tcase_add_test(tc, list_test_foreach);
	tcase_add_test(tc, strverscmp_test);
	tcase_add_test(tc, streq_test);
	tcase_add_test(tc, strneq_test);
	tcase_add_test(tc, trunkname_test);
	tcase_add_test(tc, basename_test);

	suite_add_tcase(s, tc);

	return s;
}
```

---

## Pair 18 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `dv_train_41611`
- **Project:** `brotli`
- **Commit ID:** `223d80cfbec8fd346e32906c732c8ede21f0cea6`
- **CWE:** `CWE-120`
- **Tokens:** 35

```c
static BROTLI_INLINE uint32_t GetInsertExtra(uint16_t inscode) {
  return kInsExtra[inscode];
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_50270`
- **Project:** `brotli`
- **Commit ID:** `223d80cfbec8fd346e32906c732c8ede21f0cea6`
- **Tokens:** 38

```c
static BROTLI_INLINE uint32_t GetCopyBase(uint16_t copycode) {
  return kBrotliCopyBase[copycode];
}
```

---

## Pair 19 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `primevul_train_train:936_vuln`
- **Project:** `linux`
- **Commit ID:** `b5e2f339865fb443107e5b10603e53bbc92dc054`
- **CWE:** `None`
- **Tokens:** 263

```c
int wvlan_set_station_nickname(struct net_device *dev,
		      struct iw_request_info *info,
		      union iwreq_data *wrqu,
		      char *extra)
 {
 	struct wl_private *lp = wl_priv(dev);
 	unsigned long flags;
 	int         ret = 0;
 	/*------------------------------------------------------------------------*/


	DBG_FUNC("wvlan_set_station_nickname");
	DBG_ENTER(DbgInfo);

 	wl_lock(lp, &flags);

 	memset(lp->StationName, 0, sizeof(lp->StationName));
	memcpy(lp->StationName, extra, wrqu->data.length);

 	/* Commit the adapter parameters */
 	wl_apply(lp);
	wl_unlock(lp, &flags);

	DBG_LEAVE(DbgInfo);
	return ret;
} /* wvlan_set_station_nickname */
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `primevul_train_train:936_fixed`
- **Project:** `linux`
- **Commit ID:** `b5e2f339865fb443107e5b10603e53bbc92dc054`
- **Tokens:** 241

```c
int wvlan_set_netname(struct net_device *dev,
		      struct iw_request_info *info,
		      union iwreq_data *wrqu,
		      char *extra)
{
	struct wl_private *lp = wl_priv(dev);
	unsigned long flags;
	int ret = 0;
	/*------------------------------------------------------------------------*/


	DBG_FUNC("wvlan_set_netname");
	DBG_ENTER(DbgInfo);

	wl_lock(lp, &flags);

	memset(lp->NetworkName, 0, sizeof(lp->NetworkName));
	memcpy(lp->NetworkName, extra, wrqu->data.length);

	/* Commit the adapter parameters */
	wl_apply(lp);
	wl_unlock(lp, &flags);

	DBG_LEAVE(DbgInfo);
	return ret;
} /* wvlan_set_netname */
```

---

## Pair 20 (Group Size: 3)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `dv_train_112940`
- **Project:** `zziplib`
- **Commit ID:** `596d9dfce2624e849417d4301e8d67935608aa5e`
- **CWE:** `CWE-119`
- **Tokens:** 40

```c
zzip_mem_disk_findfirst(ZZIP_MEM_DISK* dir) {
    return zzip_disk_findfirst(dir->disk); }
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_17705`
- **Project:** `gdm`
- **Commit ID:** `c25ef9245be4e0be2126ef3d075df4401949b570`
- **Tokens:** 45

```c
gdm_user_get_num_sessions (GdmUser    *user)
{
        return g_list_length (user->sessions);
}
```

---

## Pair 21 (Group Size: 3)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `dv_train_256928`
- **Project:** `electron`
- **Commit ID:** `e9fa834757f41c0b9fe44a4dffe3d7d437f52d34`
- **CWE:** `None`
- **Tokens:** 282

```c
void ElectronBrowserHandlerImpl::MessageSync(bool internal,
                                             const std::string& channel,
                                             blink::CloneableMessage arguments,
                                             MessageSyncCallback callback) {
  api::WebContents* api_web_contents = api::WebContents::From(web_contents());
  if (api_web_contents) {
    api_web_contents->MessageSync(internal, channel, std::move(arguments),
                                  std::move(callback), GetRenderFrameHost());
  }
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_7739`
- **Project:** `electron`
- **Commit ID:** `e9fa834757f41c0b9fe44a4dffe3d7d437f52d34`
- **Tokens:** 283

```c
void ElectronApiIPCHandlerImpl::MessageSync(bool internal,
                                            const std::string& channel,
                                            blink::CloneableMessage arguments,
                                            MessageSyncCallback callback) {
  api::WebContents* api_web_contents = api::WebContents::From(web_contents());
  if (api_web_contents) {
    api_web_contents->MessageSync(internal, channel, std::move(arguments),
                                  std::move(callback), GetRenderFrameHost());
  }
}
```

---

## Pair 22 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `dv_train_48001`
- **Project:** `LibRaw`
- **Commit ID:** `e67a9862d10ebaa97712f532eca1eb5e2e410a22`
- **CWE:** `CWE-400`
- **Tokens:** 369

```c
void CLASS parse_sinar_ia()
{
  int entries, off;
  char str[8], *cp;

  order = 0x4949;
  fseek(ifp, 4, SEEK_SET);
  entries = get4();
  fseek(ifp, get4(), SEEK_SET);
  while (entries--)
  {
    off = get4();
    get4();
    fread(str, 8, 1, ifp);
    if (!strcmp(str, "META"))
      meta_offset = off;
    if (!strcmp(str, "THUMB"))
      thumb_offset = off;
    if (!strcmp(str, "RAW0"))
      data_offset = off;
  }
  fseek(ifp, meta_offset + 20, SEEK_SET);
  fread(make, 64, 1, ifp);
  make[63] = 0;
  if ((cp = strchr(make, ' ')))
  {
    strcpy(model, cp + 1);
    *cp = 0;
  }
  raw_width = get2();
  raw_height = get2();
  load_raw = &CLASS unpacked_load_raw;
  thumb_width = (get4(), get2());
  thumb_height = get2();
  write_thumb = &CLASS ppm_thumb;
  maximum = 0x3fff;
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `bv_train_68795`
- **Project:** `rawstudio`
- **Commit ID:** `983bda1f0fa5fa86884381208274198a620f006e`
- **Tokens:** 344

```c
void CLASS parse_sinar_ia()
{
  int entries, off;
  char str[8], *cp;

  order = 0x4949;
  fseek (ifp, 4, SEEK_SET);
  entries = get4();
  fseek (ifp, get4(), SEEK_SET);
  while (entries--) {
    off = get4(); get4();
    fread (str, 8, 1, ifp);
    if (!strcmp(str,"META"))   meta_offset = off;
    if (!strcmp(str,"THUMB")) thumb_offset = off;
    if (!strcmp(str,"RAW0"))   data_offset = off;
  }
  fseek (ifp, meta_offset+20, SEEK_SET);
  fread (make, 64, 1, ifp);
  make[63] = 0;
  if ((cp = strchr(make,' '))) {
    strcpy (model, cp+1);
    *cp = 0;
  }
  raw_width  = get2();
  raw_height = get2();
  load_raw = &CLASS unpacked_load_raw;
  thumb_width = (get4(),get2());
  thumb_height = get2();
  write_thumb = &CLASS ppm_thumb;
  maximum = 0x3fff;
}
```

---

## Pair 23 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `dv_train_186643`
- **Project:** `linux`
- **Commit ID:** `ad67b74d2469d9b82aaa572d76474c95bc484d57`
- **CWE:** `CWE-200`
- **Tokens:** 10

```c
kernel_ptr(void)
{
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_39469`
- **Project:** `w3m`
- **Commit ID:** `18dcbadf2771cdb0c18509b14e4e73505b242753`
- **Tokens:** 9

```c
pcmap(void)
{
}
```

---

## Pair 24 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `bv_train_16814`
- **Project:** `jasper`
- **Commit ID:** `d42b2388f7f8e0332c846675133acea151fc557a`
- **CWE:** `CWE-190`
- **Tokens:** 974

```c
jpc_ms_t *jpc_getms(jas_stream_t *in, jpc_cstate_t *cstate)
{
	jpc_ms_t *ms;
	jpc_mstabent_t *mstabent;
	jas_stream_t *tmpstream;

	if (!(ms = jpc_ms_create(0))) {
		return 0;
	}

	/* Get the marker type. */
	if (jpc_getuint16(in, &ms->id) || ms->id < JPC_MS_MIN ||
	  ms->id > JPC_MS_MAX) {
		jpc_ms_destroy(ms);
		return 0;
	}

	mstabent = jpc_mstab_lookup(ms->id);
	ms->ops = &mstabent->ops;

	/* Get the marker segment length and parameters if present. */
	/* Note: It is tacitly assumed that a marker segment cannot have
	  parameters unless it has a length field.  That is, there cannot
	  be a parameters field without a length field and vice versa. */
	if (JPC_MS_HASPARMS(ms->id)) {
		/* Get the length of the marker segment. */
		if (jpc_getuint16(in, &ms->len) || ms->len < 3) {
			jpc_ms_destroy(ms);
			return 0;
		}
		/* Calculate the length of the marker segment parameters. */
		ms->len -= 2;
		/* Create and prepare a temporary memory stream from which to
		  read the marker segment parameters. */
		/* Note: This approach provides a simple way of ensuring that
		  we never read beyond the end of the marker segment (even if
		  the marker segment length is errantly set too small). */
		if (!(tmpstream = jas_stream_memopen(0, 0))) {
			jpc_ms_destroy(ms);
			return 0;
		}
		if (jas_stream_copy(tmpstream, in, ms->len) ||
		  jas_stream_seek(tmpstream, 0, SEEK_SET) < 0) {
			jas_stream_close(tmpstream);
			jpc_ms_destroy(ms);
			return 0;
		}
		/* Get the marker segment parameters. */
		if ((*ms->ops->getparms)(ms, cstate, tmpstream)) {
			ms->ops = 0;
			jpc_ms_destroy(ms);
			jas_stream_close(tmpstream);
			return 0;
		}

		if (jas_getdbglevel() > 0) {
 			jpc_ms_dump(ms, stderr);
 		}

		if (JAS_CAST(ulong, jas_stream_tell(tmpstream)) != ms->len) {
 			jas_eprintf(
 			  "warning: trailing garbage in marker segment (%ld bytes)\n",
 			  ms->len - jas_stream_tell(tmpstream));
		}

		/* Close the temporary stream. */
		jas_stream_close(tmpstream);

	} else {
		/* There are no marker segment parameters. */
		ms->len = 0;

		if (jas_getdbglevel() > 0) {
			jpc_ms_dump(ms, stderr);
		}
	}

	/* Update the code stream state information based on the type of
	  marker segment read. */
	/* Note: This is a bit of a hack, but I'm not going to define another
	  type of virtual function for this one special case. */
	if (ms->id == JPC_MS_SIZ) {
		cstate->numcomps = ms->parms.siz.numcomps;
	}

	return ms;
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_127528`
- **Project:** `jasper`
- **Commit ID:** `d42b2388f7f8e0332c846675133acea151fc557a`
- **Tokens:** 971

```c
jpc_ms_t *jpc_getms(jas_stream_t *in, jpc_cstate_t *cstate)
{
	jpc_ms_t *ms;
	jpc_mstabent_t *mstabent;
	jas_stream_t *tmpstream;

	if (!(ms = jpc_ms_create(0))) {
		return 0;
	}

	/* Get the marker type. */
	if (jpc_getuint16(in, &ms->id) || ms->id < JPC_MS_MIN ||
	  ms->id > JPC_MS_MAX) {
		jpc_ms_destroy(ms);
		return 0;
	}

	mstabent = jpc_mstab_lookup(ms->id);
	ms->ops = &mstabent->ops;

	/* Get the marker segment length and parameters if present. */
	/* Note: It is tacitly assumed that a marker segment cannot have
	  parameters unless it has a length field.  That is, there cannot
	  be a parameters field without a length field and vice versa. */
	if (JPC_MS_HASPARMS(ms->id)) {
		/* Get the length of the marker segment. */
		if (jpc_getuint16(in, &ms->len) || ms->len < 3) {
			jpc_ms_destroy(ms);
			return 0;
		}
		/* Calculate the length of the marker segment parameters. */
		ms->len -= 2;
		/* Create and prepare a temporary memory stream from which to
		  read the marker segment parameters. */
		/* Note: This approach provides a simple way of ensuring that
		  we never read beyond the end of the marker segment (even if
		  the marker segment length is errantly set too small). */
		if (!(tmpstream = jas_stream_memopen(0, 0))) {
			jpc_ms_destroy(ms);
			return 0;
		}
		if (jas_stream_copy(tmpstream, in, ms->len) ||
		  jas_stream_seek(tmpstream, 0, SEEK_SET) < 0) {
			jas_stream_close(tmpstream);
			jpc_ms_destroy(ms);
			return 0;
		}
		/* Get the marker segment parameters. */
		if ((*ms->ops->getparms)(ms, cstate, tmpstream)) {
			ms->ops = 0;
			jpc_ms_destroy(ms);
			jas_stream_close(tmpstream);
			return 0;
		}

		if (jas_getdbglevel() > 0) {
			jpc_ms_dump(ms, stderr);
		}

		if (JAS_CAST(jas_ulong, jas_stream_tell(tmpstream)) != ms->len) {
			jas_eprintf(
			  "warning: trailing garbage in marker segment (%ld bytes)\n",
			  ms->len - jas_stream_tell(tmpstream));
		}

		/* Close the temporary stream. */
		jas_stream_close(tmpstream);

	} else {
		/* There are no marker segment parameters. */
		ms->len = 0;

		if (jas_getdbglevel() > 0) {
			jpc_ms_dump(ms, stderr);
		}
	}

	/* Update the code stream state information based on the type of
	  marker segment read. */
	/* Note: This is a bit of a hack, but I'm not going to define another
	  type of virtual function for this one special case. */
	if (ms->id == JPC_MS_SIZ) {
		cstate->numcomps = ms->parms.siz.numcomps;
	}

	return ms;
}
```

---

## Pair 25 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `dv_train_181417`
- **Project:** `gimp`
- **Commit ID:** `323ecb73f7bf36788fb7066eb2d6678830cd5de7`
- **CWE:** `None`
- **Tokens:** 160

```c
ico_read_init (FILE *fp)
{
  IcoFileHeader header;
  /* read and check file header */
  if (!ico_read_int16 (fp, &header.reserved, 1)
      || !ico_read_int16 (fp, &header.resource_type, 1)
      || !ico_read_int16 (fp, &header.icon_count, 1)
      || header.reserved != 0
      || header.resource_type != 1)
    {
      return 0;
    }
  return header.icon_count;
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_228116`
- **Project:** `gimp`
- **Commit ID:** `323ecb73f7bf36788fb7066eb2d6678830cd5de7`
- **Tokens:** 171

```c
ico_read_init (FILE *fp)
{
  IcoFileHeader header;

  /* read and check file header */
  if (! ico_read_int16 (fp, &header.reserved, 1)      ||
      ! ico_read_int16 (fp, &header.resource_type, 1) ||
      ! ico_read_int16 (fp, &header.icon_count, 1)    ||
      header.reserved != 0 ||
      header.resource_type != 1)
    {
      return 0;
    }

  return header.icon_count;
}
```

---

## Pair 26 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `primevul_train_train:1044_vuln`
- **Project:** `openvpn`
- **Commit ID:** `11d21349a4e7e38a025849479b36ace7c2eec2ee`
- **CWE:** `None`
- **Tokens:** 1995

```c
openvpn_decrypt (struct buffer *buf, struct buffer work,
		 const struct crypto_options *opt,
		 const struct frame* frame)
{
  static const char error_prefix[] = "Authenticate/Decrypt packet error";
  struct gc_arena gc;
  gc_init (&gc);

  if (buf->len > 0 && opt->key_ctx_bi)
    {
      struct key_ctx *ctx = &opt->key_ctx_bi->decrypt;
      struct packet_id_net pin;
      bool have_pin = false;

      /* Verify the HMAC */
      if (ctx->hmac)
	{
	  int hmac_len;
	  uint8_t local_hmac[MAX_HMAC_KEY_LENGTH]; /* HMAC of ciphertext computed locally */

	  hmac_ctx_reset(ctx->hmac);

	  /* Assume the length of the input HMAC */
	  hmac_len = hmac_ctx_size (ctx->hmac);

	  /* Authentication fails if insufficient data in packet for HMAC */
	  if (buf->len < hmac_len)
	    CRYPT_ERROR ("missing authentication info");

	  hmac_ctx_update (ctx->hmac, BPTR (buf) + hmac_len, BLEN (buf) - hmac_len);
 	  hmac_ctx_final (ctx->hmac, local_hmac);

 	  /* Compare locally computed HMAC with packet HMAC */
	  if (memcmp (local_hmac, BPTR (buf), hmac_len))
 	    CRYPT_ERROR ("packet HMAC authentication failed");

 	  ASSERT (buf_advance (buf, hmac_len));
	}

      /* Decrypt packet ID + payload */

      if (ctx->cipher)
	{
	  const unsigned int mode = cipher_ctx_mode (ctx->cipher);
	  const int iv_size = cipher_ctx_iv_length (ctx->cipher);
	  uint8_t iv_buf[OPENVPN_MAX_IV_LENGTH];
	  int outlen;

	  /* initialize work buffer with FRAME_HEADROOM bytes of prepend capacity */
	  ASSERT (buf_init (&work, FRAME_HEADROOM_ADJ (frame, FRAME_HEADROOM_MARKER_DECRYPT)));

	  /* use IV if user requested it */
	  CLEAR (iv_buf);
	  if (opt->flags & CO_USE_IV)
	    {
	      if (buf->len < iv_size)
		CRYPT_ERROR ("missing IV info");
	      memcpy (iv_buf, BPTR (buf), iv_size);
	      ASSERT (buf_advance (buf, iv_size));
	    }

	  /* show the IV's initial state */
	  if (opt->flags & CO_USE_IV)
	    dmsg (D_PACKET_CONTENT, "DECRYPT IV: %s", format_hex (iv_buf, iv_size, 0, &gc));

	  if (buf->len < 1)
	    CRYPT_ERROR ("missing payload");

	  /* ctx->cipher was already initialized with key & keylen */
	  if (!cipher_ctx_reset (ctx->cipher, iv_buf))
	    CRYPT_ERROR ("cipher init failed");

	  /* Buffer overflow check (should never happen) */
	  if (!buf_safe (&work, buf->len))
	    CRYPT_ERROR ("buffer overflow");

	  /* Decrypt packet ID, payload */
	  if (!cipher_ctx_update (ctx->cipher, BPTR (&work), &outlen, BPTR (buf), BLEN (buf)))
	    CRYPT_ERROR ("cipher update failed");
	  work.len += outlen;

	  /* Flush the decryption buffer */
	  if (!cipher_ctx_final (ctx->cipher, BPTR (&work) + outlen, &outlen))
	    CRYPT_ERROR ("cipher final failed");
	  work.len += outlen;

	  dmsg (D_PACKET_CONTENT, "DECRYPT TO: %s",
	       format_hex (BPTR (&work), BLEN (&work), 80, &gc));

	  /* Get packet ID from plaintext buffer or IV, depending on cipher mode */
	  {
	    if (mode == OPENVPN_MODE_CBC)
	      {
		if (opt->packet_id)
		  {
		    if (!packet_id_read (&pin, &work, BOOL_CAST (opt->flags & CO_PACKET_ID_LONG_FORM)))
		      CRYPT_ERROR ("error reading CBC packet-id");
		    have_pin = true;
		  }
	      }
	    else if (mode == OPENVPN_MODE_CFB || mode == OPENVPN_MODE_OFB)
	      {
		struct buffer b;

		ASSERT (opt->flags & CO_USE_IV);    /* IV and packet-ID required */
		ASSERT (opt->packet_id); /*  for this mode. */

		buf_set_read (&b, iv_buf, iv_size);
		if (!packet_id_read (&pin, &b, true))
		  CRYPT_ERROR ("error reading CFB/OFB packet-id");
		have_pin = true;
	      }
	    else /* We only support CBC, CFB, or OFB modes right now */
	      {
		ASSERT (0);
	      }
	  }
	}
      else
	{
	  work = *buf;
	  if (opt->packet_id)
	    {
	      if (!packet_id_read (&pin, &work, BOOL_CAST (opt->flags & CO_PACKET_ID_LONG_FORM)))
		CRYPT_ERROR ("error reading packet-id");
	      have_pin = !BOOL_CAST (opt->flags & CO_IGNORE_PACKET_ID);
	    }
	}

      if (have_pin)
	{
	  packet_id_reap_test (&opt->packet_id->rec);
	  if (packet_id_test (&opt->packet_id->rec, &pin))
	    {
	      packet_id_add (&opt->packet_id->rec, &pin);
	      if (opt->pid_persist && (opt->flags & CO_PACKET_ID_LONG_FORM))
		packet_id_persist_save_obj (opt->pid_persist, opt->packet_id);
	    }
	  else
	    {
	      if (!(opt->flags & CO_MUTE_REPLAY_WARNINGS))
	      msg (D_REPLAY_ERRORS, "%s: bad packet ID (may be a replay): %s -- see the man page entry for --no-replay and --replay-window for more info or silence this warning with --mute-replay-warnings",
		   error_prefix, packet_id_net_print (&pin, true, &gc));
	      goto error_exit;
	    }
	}
      *buf = work;
    }

  gc_free (&gc);
  return true;

 error_exit:
  crypto_clear_error();
  buf->len = 0;
  gc_free (&gc);
  return false;
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_16618`
- **Project:** `openvpn`
- **Commit ID:** `11d21349a4e7e38a025849479b36ace7c2eec2ee`
- **Tokens:** 1996

```c
openvpn_decrypt (struct buffer *buf, struct buffer work,
		 const struct crypto_options *opt,
		 const struct frame* frame)
{
  static const char error_prefix[] = "Authenticate/Decrypt packet error";
  struct gc_arena gc;
  gc_init (&gc);

  if (buf->len > 0 && opt->key_ctx_bi)
    {
      struct key_ctx *ctx = &opt->key_ctx_bi->decrypt;
      struct packet_id_net pin;
      bool have_pin = false;

      /* Verify the HMAC */
      if (ctx->hmac)
	{
	  int hmac_len;
	  uint8_t local_hmac[MAX_HMAC_KEY_LENGTH]; /* HMAC of ciphertext computed locally */

	  hmac_ctx_reset(ctx->hmac);

	  /* Assume the length of the input HMAC */
	  hmac_len = hmac_ctx_size (ctx->hmac);

	  /* Authentication fails if insufficient data in packet for HMAC */
	  if (buf->len < hmac_len)
	    CRYPT_ERROR ("missing authentication info");

	  hmac_ctx_update (ctx->hmac, BPTR (buf) + hmac_len, BLEN (buf) - hmac_len);
	  hmac_ctx_final (ctx->hmac, local_hmac);

	  /* Compare locally computed HMAC with packet HMAC */
	  if (memcmp_constant_time (local_hmac, BPTR (buf), hmac_len))
	    CRYPT_ERROR ("packet HMAC authentication failed");

	  ASSERT (buf_advance (buf, hmac_len));
	}

      /* Decrypt packet ID + payload */

      if (ctx->cipher)
	{
	  const unsigned int mode = cipher_ctx_mode (ctx->cipher);
	  const int iv_size = cipher_ctx_iv_length (ctx->cipher);
	  uint8_t iv_buf[OPENVPN_MAX_IV_LENGTH];
	  int outlen;

	  /* initialize work buffer with FRAME_HEADROOM bytes of prepend capacity */
	  ASSERT (buf_init (&work, FRAME_HEADROOM_ADJ (frame, FRAME_HEADROOM_MARKER_DECRYPT)));

	  /* use IV if user requested it */
	  CLEAR (iv_buf);
	  if (opt->flags & CO_USE_IV)
	    {
	      if (buf->len < iv_size)
		CRYPT_ERROR ("missing IV info");
	      memcpy (iv_buf, BPTR (buf), iv_size);
	      ASSERT (buf_advance (buf, iv_size));
	    }

	  /* show the IV's initial state */
	  if (opt->flags & CO_USE_IV)
	    dmsg (D_PACKET_CONTENT, "DECRYPT IV: %s", format_hex (iv_buf, iv_size, 0, &gc));

	  if (buf->len < 1)
	    CRYPT_ERROR ("missing payload");

	  /* ctx->cipher was already initialized with key & keylen */
	  if (!cipher_ctx_reset (ctx->cipher, iv_buf))
	    CRYPT_ERROR ("cipher init failed");

	  /* Buffer overflow check (should never happen) */
	  if (!buf_safe (&work, buf->len))
	    CRYPT_ERROR ("buffer overflow");

	  /* Decrypt packet ID, payload */
	  if (!cipher_ctx_update (ctx->cipher, BPTR (&work), &outlen, BPTR (buf), BLEN (buf)))
	    CRYPT_ERROR ("cipher update failed");
	  work.len += outlen;

	  /* Flush the decryption buffer */
	  if (!cipher_ctx_final (ctx->cipher, BPTR (&work) + outlen, &outlen))
	    CRYPT_ERROR ("cipher final failed");
	  work.len += outlen;

	  dmsg (D_PACKET_CONTENT, "DECRYPT TO: %s",
	       format_hex (BPTR (&work), BLEN (&work), 80, &gc));

	  /* Get packet ID from plaintext buffer or IV, depending on cipher mode */
	  {
	    if (mode == OPENVPN_MODE_CBC)
	      {
		if (opt->packet_id)
		  {
		    if (!packet_id_read (&pin, &work, BOOL_CAST (opt->flags & CO_PACKET_ID_LONG_FORM)))
		      CRYPT_ERROR ("error reading CBC packet-id");
		    have_pin = true;
		  }
	      }
	    else if (mode == OPENVPN_MODE_CFB || mode == OPENVPN_MODE_OFB)
	      {
		struct buffer b;

		ASSERT (opt->flags & CO_USE_IV);    /* IV and packet-ID required */
		ASSERT (opt->packet_id); /*  for this mode. */

		buf_set_read (&b, iv_buf, iv_size);
		if (!packet_id_read (&pin, &b, true))
		  CRYPT_ERROR ("error reading CFB/OFB packet-id");
		have_pin = true;
	      }
	    else /* We only support CBC, CFB, or OFB modes right now */
	      {
		ASSERT (0);
	      }
	  }
	}
      else
	{
	  work = *buf;
	  if (opt->packet_id)
	    {
	      if (!packet_id_read (&pin, &work, BOOL_CAST (opt->flags & CO_PACKET_ID_LONG_FORM)))
		CRYPT_ERROR ("error reading packet-id");
	      have_pin = !BOOL_CAST (opt->flags & CO_IGNORE_PACKET_ID);
	    }
	}

      if (have_pin)
	{
	  packet_id_reap_test (&opt->packet_id->rec);
	  if (packet_id_test (&opt->packet_id->rec, &pin))
	    {
	      packet_id_add (&opt->packet_id->rec, &pin);
	      if (opt->pid_persist && (opt->flags & CO_PACKET_ID_LONG_FORM))
		packet_id_persist_save_obj (opt->pid_persist, opt->packet_id);
	    }
	  else
	    {
	      if (!(opt->flags & CO_MUTE_REPLAY_WARNINGS))
	      msg (D_REPLAY_ERRORS, "%s: bad packet ID (may be a replay): %s -- see the man page entry for --no-replay and --replay-window for more info or silence this warning with --mute-replay-warnings",
		   error_prefix, packet_id_net_print (&pin, true, &gc));
	      goto error_exit;
	    }
	}
      *buf = work;
    }

  gc_free (&gc);
  return true;

 error_exit:
  crypto_clear_error();
  buf->len = 0;
  gc_free (&gc);
  return false;
}
```

---

## Pair 27 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `dv_train_250145`
- **Project:** `heimdal`
- **Commit ID:** `b1e699103f08d6a0ca46a122193c9da65f6cf837`
- **CWE:** `CWE-284`
- **Tokens:** 1318

```c
fix_transited_encoding(krb5_context context,
		       krb5_kdc_configuration *config,
		       krb5_boolean check_policy,
		       const TransitedEncoding *tr,
		       EncTicketPart *et,
		       const char *client_realm,
		       const char *server_realm,
		       const char *tgt_realm)
{
    krb5_error_code ret = 0;
    char **realms, **tmp;
    unsigned int num_realms;
    size_t i;

    switch (tr->tr_type) {
    case DOMAIN_X500_COMPRESS:
	break;
    case 0:
	/*
	 * Allow empty content of type 0 because that is was Microsoft
	 * generates in their TGT.
	 */
	if (tr->contents.length == 0)
	    break;
	kdc_log(context, config, 0,
		"Transited type 0 with non empty content");
	return KRB5KDC_ERR_TRTYPE_NOSUPP;
    default:
	kdc_log(context, config, 0,
		"Unknown transited type: %u", tr->tr_type);
	return KRB5KDC_ERR_TRTYPE_NOSUPP;
    }

    ret = krb5_domain_x500_decode(context,
				  tr->contents,
				  &realms,
				  &num_realms,
				  client_realm,
				  server_realm);
    if(ret){
	krb5_warn(context, ret,
		  "Decoding transited encoding");
	return ret;
    }
    if(strcmp(client_realm, tgt_realm) && strcmp(server_realm, tgt_realm)) {
	/* not us, so add the previous realm to transited set */
	if (num_realms + 1 > UINT_MAX/sizeof(*realms)) {
	    ret = ERANGE;
	    goto free_realms;
	}
	tmp = realloc(realms, (num_realms + 1) * sizeof(*realms));
	if(tmp == NULL){
	    ret = ENOMEM;
	    goto free_realms;
	}
	realms = tmp;
	realms[num_realms] = strdup(tgt_realm);
	if(realms[num_realms] == NULL){
	    ret = ENOMEM;
	    goto free_realms;
	}
	num_realms++;
    }
    if(num_realms == 0) {
	if(strcmp(client_realm, server_realm))
	    kdc_log(context, config, 0,
		    "cross-realm %s -> %s", client_realm, server_realm);
    } else {
	size_t l = 0;
	char *rs;
	for(i = 0; i < num_realms; i++)
	    l += strlen(realms[i]) + 2;
	rs = malloc(l);
	if(rs != NULL) {
	    *rs = '\0';
	    for(i = 0; i < num_realms; i++) {
		if(i > 0)
		    strlcat(rs, ", ", l);
		strlcat(rs, realms[i], l);
	    }
	    kdc_log(context, config, 0,
		    "cross-realm %s -> %s via [%s]",
		    client_realm, server_realm, rs);
	    free(rs);
	}
    }
    if(check_policy) {
	ret = krb5_check_transited(context, client_realm,
				   server_realm,
				   realms, num_realms, NULL);
	if(ret) {
	    krb5_warn(context, ret, "cross-realm %s -> %s",
		      client_realm, server_realm);
	    goto free_realms;
	}
	et->flags.transited_policy_checked = 1;
    }
    et->transited.tr_type = DOMAIN_X500_COMPRESS;
    ret = krb5_domain_x500_encode(realms, num_realms, &et->transited.contents);
    if(ret)
	krb5_warn(context, ret, "Encoding transited encoding");
  free_realms:
    for(i = 0; i < num_realms; i++)
	free(realms[i]);
    free(realms);
    return ret;
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_126231`
- **Project:** `heimdal`
- **Commit ID:** `04171147948d0a3636bc6374181926f0fb2ec83a`
- **Tokens:** 1354

```c
fix_transited_encoding(krb5_context context,
		       krb5_kdc_configuration *config,
		       krb5_boolean check_policy,
		       const TransitedEncoding *tr,
		       EncTicketPart *et,
		       const char *client_realm,
		       const char *server_realm,
		       const char *tgt_realm)
{
    krb5_error_code ret = 0;
    char **realms, **tmp;
    unsigned int num_realms;
    size_t i;

    switch (tr->tr_type) {
    case domain_X500_Compress:
	break;
    case 0:
	/*
	 * Allow empty content of type 0 because that is was Microsoft
	 * generates in their TGT.
	 */
	if (tr->contents.length == 0)
	    break;
	kdc_log(context, config, 4,
		"Transited type 0 with non empty content");
	return KRB5KDC_ERR_TRTYPE_NOSUPP;
    default:
	kdc_log(context, config, 4,
		"Unknown transited type: %u", tr->tr_type);
	return KRB5KDC_ERR_TRTYPE_NOSUPP;
    }

    ret = krb5_domain_x500_decode(context,
				  tr->contents,
				  &realms,
				  &num_realms,
				  client_realm,
				  server_realm);
    if(ret){
	krb5_warn(context, ret,
		  "Decoding transited encoding");
	return ret;
    }

    /*
     * If the realm of the presented tgt is neither the client nor the server
     * realm, it is a transit realm and must be added to transited set.
     */
    if(strcmp(client_realm, tgt_realm) && strcmp(server_realm, tgt_realm)) {
	if (num_realms + 1 > UINT_MAX/sizeof(*realms)) {
	    ret = ERANGE;
	    goto free_realms;
	}
	tmp = realloc(realms, (num_realms + 1) * sizeof(*realms));
	if(tmp == NULL){
	    ret = ENOMEM;
	    goto free_realms;
	}
	realms = tmp;
	realms[num_realms] = strdup(tgt_realm);
	if(realms[num_realms] == NULL){
	    ret = ENOMEM;
	    goto free_realms;
	}
	num_realms++;
    }
    if(num_realms == 0) {
	if(strcmp(client_realm, server_realm))
	    kdc_log(context, config, 4,
		    "cross-realm %s -> %s", client_realm, server_realm);
    } else {
	size_t l = 0;
	char *rs;
	for(i = 0; i < num_realms; i++)
	    l += strlen(realms[i]) + 2;
	rs = malloc(l);
	if(rs != NULL) {
	    *rs = '\0';
	    for(i = 0; i < num_realms; i++) {
		if(i > 0)
		    strlcat(rs, ", ", l);
		strlcat(rs, realms[i], l);
	    }
	    kdc_log(context, config, 4,
		    "cross-realm %s -> %s via [%s]",
		    client_realm, server_realm, rs);
	    free(rs);
	}
    }
    if(check_policy) {
	ret = krb5_check_transited(context, client_realm,
				   server_realm,
				   realms, num_realms, NULL);
	if(ret) {
	    krb5_warn(context, ret, "cross-realm %s -> %s",
		      client_realm, server_realm);
	    goto free_realms;
	}
	et->flags.transited_policy_checked = 1;
    }
    et->transited.tr_type = domain_X500_Compress;
    ret = krb5_domain_x500_encode(realms, num_realms, &et->transited.contents);
    if(ret)
	krb5_warn(context, ret, "Encoding transited encoding");
  free_realms:
    for(i = 0; i < num_realms; i++)
	free(realms[i]);
    free(realms);
    return ret;
}
```

---

## Pair 28 (Group Size: 3)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `bv_train_89653`
- **Project:** `php`
- **Commit ID:** `dccda88f27a084bcbbb30198ace12b4e7ae961cc`
- **CWE:** `CWE-119`
- **Tokens:** 52

```c
PHP_FUNCTION(xml_parser_create)
 {
       php_xml_parser_create_impl(INTERNAL_FUNCTION_PARAM_PASSTHRU, 0);
 }
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_5100`
- **Project:** `php-src`
- **Commit ID:** `2938329ce19cb8c4197dec146c3ec887c6f61d01`
- **Tokens:** 42

```c
PHP_FUNCTION(imagestringup)
{
	php_imagechar(INTERNAL_FUNCTION_PARAM_PASSTHRU, 3);
}
```

---

## Pair 29 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `primevul_train_train:209073_vuln`
- **Project:** `zziplib`
- **Commit ID:** `feae4da1a5c92100c44ebfcbaaa895959cc0829b`
- **CWE:** `None`
- **Tokens:** 3622

```c
__zzip_parse_root_directory(int fd,
                            struct _disk_trailer *trailer,
                            struct zzip_dir_hdr **hdr_return,
                            zzip_plugin_io_t io)
{
    auto struct zzip_disk_entry dirent;
    struct zzip_dir_hdr *hdr;
    struct zzip_dir_hdr *hdr0;
    uint16_t *p_reclen = 0;
    zzip_off64_t entries;
    zzip_off64_t zz_offset;     /* offset from start of root directory */
    char *fd_map = 0;
    zzip_off64_t zz_fd_gap = 0;
    zzip_off64_t zz_entries = _disk_trailer_localentries(trailer);
    zzip_off64_t zz_rootsize = _disk_trailer_rootsize(trailer);
    zzip_off64_t zz_rootseek = _disk_trailer_rootseek(trailer);
    __correct_rootseek(zz_rootseek, zz_rootsize, trailer);

    if (zz_entries < 0 || zz_rootseek < 0 || zz_rootseek < 0)
        return ZZIP_CORRUPTED;

    hdr0 = (struct zzip_dir_hdr *) malloc(zz_rootsize);
    if (! hdr0)
        return ZZIP_DIRSIZE;
    hdr = hdr0;
    __debug_dir_hdr(hdr);

    if (USE_MMAP && io->fd.sys)
    {
        zz_fd_gap = zz_rootseek & (_zzip_getpagesize(io->fd.sys) - 1);
        HINT4(" fd_gap=%ld, mapseek=0x%lx, maplen=%ld", (long) (zz_fd_gap),
              (long) (zz_rootseek - zz_fd_gap),
              (long) (zz_rootsize + zz_fd_gap));
        fd_map =
            _zzip_mmap(io->fd.sys, fd, zz_rootseek - zz_fd_gap,
                       zz_rootsize + zz_fd_gap);
        /* if mmap failed we will fallback to seek/read mode */
        if (fd_map == MAP_FAILED)
        {
            NOTE2("map failed: %s", strerror(errno));
            fd_map = 0;
        } else
        {
            HINT3("mapped *%p len=%li", fd_map,
                  (long) (zz_rootsize + zz_fd_gap));
        }
    }

    for (entries=0, zz_offset=0; ; entries++)
    {
        register struct zzip_disk_entry *d;
        uint16_t u_extras, u_comment, u_namlen;

#     ifndef ZZIP_ALLOW_MODULO_ENTRIES
        if (entries >= zz_entries) {
            if (zz_offset + 256 < zz_rootsize) {
                FAIL4("%li's entry is long before the end of directory - enable modulo_entries? (O:%li R:%li)",
                      (long) entries, (long) (zz_offset), (long) zz_rootsize);
            }
            break;
        }
#     endif

        if (fd_map)
        {
            d = (void*)(fd_map+zz_fd_gap+zz_offset); /* fd_map+fd_gap==u_rootseek */
        } else
        {
            if (io->fd.seeks(fd, zz_rootseek + zz_offset, SEEK_SET) < 0)
                return ZZIP_DIR_SEEK;
            if (io->fd.read(fd, &dirent, sizeof(dirent)) < __sizeof(dirent))
                return ZZIP_DIR_READ;
            d = &dirent;
        }

        if ((zzip_off64_t) (zz_offset + sizeof(*d)) > zz_rootsize ||
            (zzip_off64_t) (zz_offset + sizeof(*d)) < 0)
        {
            FAIL4("%li's entry stretches beyond root directory (O:%li R:%li)",
                  (long) entries, (long) (zz_offset), (long) zz_rootsize);
            break;
        }

        if (! zzip_disk_entry_check_magic(d)) {
#        ifndef ZZIP_ALLOW_MODULO_ENTRIES
            FAIL4("%li's entry has no disk_entry magic indicator (O:%li R:%li)",
                  (long) entries, (long) (zz_offset), (long) zz_rootsize);
#        endif
            break;
        }

#       if 0 && defined DEBUG
        zzip_debug_xbuf((unsigned char *) d, sizeof(*d) + 8);
#       endif

        u_extras = zzip_disk_entry_get_extras(d);
        u_comment = zzip_disk_entry_get_comment(d);
        u_namlen = zzip_disk_entry_get_namlen(d);
        HINT5("offset=0x%lx, size %ld, dirent *%p, hdr %p\n",
              (long) (zz_offset + zz_rootseek), (long) zz_rootsize, d, hdr);

        /* writes over the read buffer, Since the structure where data is
           copied is smaller than the data in buffer this can be done.
           It is important that the order of setting the fields is considered
           when filling the structure, so that some data is not trashed in
           first structure read.
           at the end the whole copied list of structures  is copied into
           newly allocated buffer */
        hdr->d_crc32 = zzip_disk_entry_get_crc32(d);
        hdr->d_csize = zzip_disk_entry_get_csize(d);
        hdr->d_usize = zzip_disk_entry_get_usize(d);
        hdr->d_off = zzip_disk_entry_get_offset(d);
        hdr->d_compr = zzip_disk_entry_get_compr(d);
        if (hdr->d_compr > _255)
            hdr->d_compr = 255;

        if ((zzip_off64_t) (zz_offset + sizeof(*d) + u_namlen) > zz_rootsize ||
            (zzip_off64_t) (zz_offset + sizeof(*d) + u_namlen) < 0)
        {
            FAIL4("%li's name stretches beyond root directory (O:%li N:%li)",
                  (long) entries, (long) (zz_offset), (long) (u_namlen));
            break;
        }

        if (fd_map)
            {  memcpy(hdr->d_name, fd_map+zz_fd_gap + zz_offset+sizeof(*d), u_namlen); }
        else
            { io->fd.read(fd, hdr->d_name, u_namlen); }
        hdr->d_name[u_namlen] = '\0';
        hdr->d_namlen = u_namlen;

        /* update offset by the total length of this entry -> next entry */
        zz_offset += sizeof(*d) + u_namlen + u_extras + u_comment;

        if (zz_offset > zz_rootsize)
        {
            FAIL3("%li's entry stretches beyond root directory (O:%li)",
                  (long) entries, (long) (zz_offset));
            entries ++;
            break;
        }

        HINT5("file %ld { compr=%d crc32=$%x offset=%d",
              (long) entries, hdr->d_compr, hdr->d_crc32, hdr->d_off);
        HINT5("csize=%d usize=%d namlen=%d extras=%d",
              hdr->d_csize, hdr->d_usize, u_namlen, u_extras);
        HINT5("comment=%d name='%s' %s <sizeof %d> } ",
              u_comment, hdr->d_name, "", (int) sizeof(*d));

        p_reclen = &hdr->d_reclen;

        {
            register char *p = (char *) hdr;
            register char *q = aligned4(p + sizeof(*hdr) + u_namlen + 1);
            *p_reclen = (uint16_t) (q - p);
            hdr = (struct zzip_dir_hdr *) q;
        }
    }                           /*for */

    if (USE_MMAP && fd_map)
    {
        HINT3("unmap *%p len=%li", fd_map, (long) (zz_rootsize + zz_fd_gap));
        _zzip_munmap(io->fd.sys, fd_map, zz_rootsize + zz_fd_gap);
    }

    if (p_reclen)
    {
        *p_reclen = 0;          /* mark end of list */

        if (hdr_return)
            *hdr_return = hdr0;
    }                           /* else zero (sane) entries */
#  ifndef ZZIP_ALLOW_MODULO_ENTRIES
    return (entries != zz_entries ? ZZIP_CORRUPTED : 0);
#  else
    return ((entries & (unsigned)0xFFFF) != zz_entries ? ZZIP_CORRUPTED : 0);
#  endif
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `primevul_train_train:209073_fixed`
- **Project:** `zziplib`
- **Commit ID:** `feae4da1a5c92100c44ebfcbaaa895959cc0829b`
- **Tokens:** 3622

```c
__zzip_parse_root_directory(int fd,
                            struct _disk_trailer *trailer,
                            struct zzip_dir_hdr **hdr_return,
                            zzip_plugin_io_t io)
{
    auto struct zzip_disk_entry dirent;
    struct zzip_dir_hdr *hdr;
    struct zzip_dir_hdr *hdr0;
    uint16_t *p_reclen = 0;
    zzip_off64_t entries;
    zzip_off64_t zz_offset;     /* offset from start of root directory */
    char *fd_map = 0;
    zzip_off64_t zz_fd_gap = 0;
    zzip_off64_t zz_entries = _disk_trailer_localentries(trailer);
    zzip_off64_t zz_rootsize = _disk_trailer_rootsize(trailer);
    zzip_off64_t zz_rootseek = _disk_trailer_rootseek(trailer);
    __correct_rootseek(zz_rootseek, zz_rootsize, trailer);

    if (zz_entries < 0 || zz_rootseek < 0 || zz_rootsize < 0)
        return ZZIP_CORRUPTED;

    hdr0 = (struct zzip_dir_hdr *) malloc(zz_rootsize);
    if (! hdr0)
        return ZZIP_DIRSIZE;
    hdr = hdr0;
    __debug_dir_hdr(hdr);

    if (USE_MMAP && io->fd.sys)
    {
        zz_fd_gap = zz_rootseek & (_zzip_getpagesize(io->fd.sys) - 1);
        HINT4(" fd_gap=%ld, mapseek=0x%lx, maplen=%ld", (long) (zz_fd_gap),
              (long) (zz_rootseek - zz_fd_gap),
              (long) (zz_rootsize + zz_fd_gap));
        fd_map =
            _zzip_mmap(io->fd.sys, fd, zz_rootseek - zz_fd_gap,
                       zz_rootsize + zz_fd_gap);
        /* if mmap failed we will fallback to seek/read mode */
        if (fd_map == MAP_FAILED)
        {
            NOTE2("map failed: %s", strerror(errno));
            fd_map = 0;
        } else
        {
            HINT3("mapped *%p len=%li", fd_map,
                  (long) (zz_rootsize + zz_fd_gap));
        }
    }

    for (entries=0, zz_offset=0; ; entries++)
    {
        register struct zzip_disk_entry *d;
        uint16_t u_extras, u_comment, u_namlen;

#     ifndef ZZIP_ALLOW_MODULO_ENTRIES
        if (entries >= zz_entries) {
            if (zz_offset + 256 < zz_rootsize) {
                FAIL4("%li's entry is long before the end of directory - enable modulo_entries? (O:%li R:%li)",
                      (long) entries, (long) (zz_offset), (long) zz_rootsize);
            }
            break;
        }
#     endif

        if (fd_map)
        {
            d = (void*)(fd_map+zz_fd_gap+zz_offset); /* fd_map+fd_gap==u_rootseek */
        } else
        {
            if (io->fd.seeks(fd, zz_rootseek + zz_offset, SEEK_SET) < 0)
                return ZZIP_DIR_SEEK;
            if (io->fd.read(fd, &dirent, sizeof(dirent)) < __sizeof(dirent))
                return ZZIP_DIR_READ;
            d = &dirent;
        }

        if ((zzip_off64_t) (zz_offset + sizeof(*d)) > zz_rootsize ||
            (zzip_off64_t) (zz_offset + sizeof(*d)) < 0)
        {
            FAIL4("%li's entry stretches beyond root directory (O:%li R:%li)",
                  (long) entries, (long) (zz_offset), (long) zz_rootsize);
            break;
        }

        if (! zzip_disk_entry_check_magic(d)) {
#        ifndef ZZIP_ALLOW_MODULO_ENTRIES
            FAIL4("%li's entry has no disk_entry magic indicator (O:%li R:%li)",
                  (long) entries, (long) (zz_offset), (long) zz_rootsize);
#        endif
            break;
        }

#       if 0 && defined DEBUG
        zzip_debug_xbuf((unsigned char *) d, sizeof(*d) + 8);
#       endif

        u_extras = zzip_disk_entry_get_extras(d);
        u_comment = zzip_disk_entry_get_comment(d);
        u_namlen = zzip_disk_entry_get_namlen(d);
        HINT5("offset=0x%lx, size %ld, dirent *%p, hdr %p\n",
              (long) (zz_offset + zz_rootseek), (long) zz_rootsize, d, hdr);

        /* writes over the read buffer, Since the structure where data is
           copied is smaller than the data in buffer this can be done.
           It is important that the order of setting the fields is considered
           when filling the structure, so that some data is not trashed in
           first structure read.
           at the end the whole copied list of structures  is copied into
           newly allocated buffer */
        hdr->d_crc32 = zzip_disk_entry_get_crc32(d);
        hdr->d_csize = zzip_disk_entry_get_csize(d);
        hdr->d_usize = zzip_disk_entry_get_usize(d);
        hdr->d_off = zzip_disk_entry_get_offset(d);
        hdr->d_compr = zzip_disk_entry_get_compr(d);
        if (hdr->d_compr > _255)
            hdr->d_compr = 255;

        if ((zzip_off64_t) (zz_offset + sizeof(*d) + u_namlen) > zz_rootsize ||
            (zzip_off64_t) (zz_offset + sizeof(*d) + u_namlen) < 0)
        {
            FAIL4("%li's name stretches beyond root directory (O:%li N:%li)",
                  (long) entries, (long) (zz_offset), (long) (u_namlen));
            break;
        }

        if (fd_map)
            {  memcpy(hdr->d_name, fd_map+zz_fd_gap + zz_offset+sizeof(*d), u_namlen); }
        else
            { io->fd.read(fd, hdr->d_name, u_namlen); }
        hdr->d_name[u_namlen] = '\0';
        hdr->d_namlen = u_namlen;

        /* update offset by the total length of this entry -> next entry */
        zz_offset += sizeof(*d) + u_namlen + u_extras + u_comment;

        if (zz_offset > zz_rootsize)
        {
            FAIL3("%li's entry stretches beyond root directory (O:%li)",
                  (long) entries, (long) (zz_offset));
            entries ++;
            break;
        }

        HINT5("file %ld { compr=%d crc32=$%x offset=%d",
              (long) entries, hdr->d_compr, hdr->d_crc32, hdr->d_off);
        HINT5("csize=%d usize=%d namlen=%d extras=%d",
              hdr->d_csize, hdr->d_usize, u_namlen, u_extras);
        HINT5("comment=%d name='%s' %s <sizeof %d> } ",
              u_comment, hdr->d_name, "", (int) sizeof(*d));

        p_reclen = &hdr->d_reclen;

        {
            register char *p = (char *) hdr;
            register char *q = aligned4(p + sizeof(*hdr) + u_namlen + 1);
            *p_reclen = (uint16_t) (q - p);
            hdr = (struct zzip_dir_hdr *) q;
        }
    }                           /*for */

    if (USE_MMAP && fd_map)
    {
        HINT3("unmap *%p len=%li", fd_map, (long) (zz_rootsize + zz_fd_gap));
        _zzip_munmap(io->fd.sys, fd_map, zz_rootsize + zz_fd_gap);
    }

    if (p_reclen)
    {
        *p_reclen = 0;          /* mark end of list */

        if (hdr_return)
            *hdr_return = hdr0;
    }                           /* else zero (sane) entries */
#  ifndef ZZIP_ALLOW_MODULO_ENTRIES
    return (entries != zz_entries ? ZZIP_CORRUPTED : 0);
#  else
    return ((entries & (unsigned)0xFFFF) != zz_entries ? ZZIP_CORRUPTED : 0);
#  endif
}
```

---

## Pair 30 (Group Size: 2)
### Vulnerable Sample (vuln_label = 1)
- **Sample ID:** `bv_train_39503`
- **Project:** `Android`
- **Commit ID:** `04839626ed859623901ebd3a5fd483982186b59d`
- **CWE:** `CWE-119`
- **Tokens:** 105

```c
SegmentInfo::~SegmentInfo()
{
    delete[] m_pMuxingAppAsUTF8;
    m_pMuxingAppAsUTF8 = NULL;
    delete[] m_pWritingAppAsUTF8;
    m_pWritingAppAsUTF8 = NULL;
    delete[] m_pTitleAsUTF8;
    m_pTitleAsUTF8 = NULL;
}
```

### Benign Sample (vuln_label = 0)
- **Sample ID:** `dv_train_60754`
- **Project:** `libvpx`
- **Commit ID:** `34d54b04e98dd0bac32e9aab0fbda0bf501bc742`
- **Tokens:** 92

```c
SegmentInfo::~SegmentInfo() {
  delete[] m_pMuxingAppAsUTF8;
  m_pMuxingAppAsUTF8 = NULL;

  delete[] m_pWritingAppAsUTF8;
  m_pWritingAppAsUTF8 = NULL;

  delete[] m_pTitleAsUTF8;
  m_pTitleAsUTF8 = NULL;
}
```

---

